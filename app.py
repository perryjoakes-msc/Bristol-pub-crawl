import os
import pickle
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

# ---------------------------------------------------------
# PAGE SETUP & CSS INJECTION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bristol Pub Crawl",
    page_icon="🍺",
    layout="wide",  # Better base defaults for mobile than 'wide'
    initial_sidebar_state="expanded",
)

# Custom Mobile CSS to eliminate horizontal scrolling & black bars
# Custom Mobile CSS to make the sidebar narrower using viewport units (vw)
st.markdown(
    """
    <style>
    /* 1. HIDE ONLY FORK/GITHUB/MENU (LEAVE SIDEBAR TOGGLE BUTTON ACTIVE) */
    [data-testid="stHeaderActionElements"], 
    [data-testid="stToolbar"],
    footer {
        display: none !important;
    }

    /* Keep header background transparent so toggle button floats cleanly */
    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 2rem !important;
    }

    /* Style the sidebar expand/collapse chevron button */
    [data-testid="stSidebarCollapseButton"] {
        color: white !important;
        visibility: visible !important;
    }

    /* 2. LOCK VIEWPORT & ELIMINATE HORIZONTAL SCROLLING */
    html, body, [data-testid="stAppViewContainer"], .main {
        overflow-x: hidden !important;
        max-width: 100vw !important;
        background-color: #0e1117 !important;
    }

    /* 3. SIDEBAR WIDTH & HARD-RESET TOP PADDING */
    [data-testid="stSidebar"] {
        width: 75vw !important;
        max-width: 320px !important;
    }

    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarContent"] {
        padding-top: 0.5rem !important;
        margin-top: 0rem !important;
    }

    /* 4. MAIN CONTAINER EDGE-TO-EDGE BLEED */
    [data-testid="stAppViewContainer"] > .main {
        width: 100% !important;
        flex: 1 1 auto !important;
    }
    
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.25rem !important;
        padding-right: 0.25rem !important;
        max-width: 100% !important;
    }

    /* 5. FORCE MAP CANVAS TO STAY IN BOUNDS */
    iframe, .stFolium {
        max-width: 100% !important;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#st.title("🍺 Bristol Pub Crawl 🍺")


# ---------------------------------------------------------
# 1. LOAD DATASETS & GRAPH
# ---------------------------------------------------------
@st.cache_data
def load_datasets():
    pubs_path = "bristol_pubs.gpkg"
    if not os.path.exists(pubs_path):
        st.error(f"Ensure '{pubs_path}' is in your current directory.")
        st.stop()

    pubs_gdf = gpd.read_file(pubs_path).fillna("NA").rename(
        columns={
            "cocktails..yes.no.": "cocktails",
            "pool_table..yes.no.": "pool_table",
            "outside_seating..yes.no.": "outside_seating",
            "name": "pub_name",
        }
    )
    if pubs_gdf.crs != 27700:
        pubs_gdf = pubs_gdf.to_crs(27700)

    pubs_gdf["density_score"] = [
        max(0, (pubs_gdf.geometry.distance(pub.geometry) <= 300.0).sum() - 1)
        for _, pub in pubs_gdf.iterrows()
    ]
    return pubs_gdf

@st.cache_resource
def load_graph():
    with open("bristol_graph.pkl", "rb") as f:
        G = pickle.load(f)
    return G

PUBS_GDF = load_datasets()
G = load_graph()

# ---------------------------------------------------------
# 2. CORE ROUTING ENGINE
# ---------------------------------------------------------
def calculate_pub_utility(pub_row, weights):
    utility = 0.0
    binary_cols = ["beer_garden", "cocktails", "pool_table"]
    for col in binary_cols:
        if col in weights:
            val = str(pub_row.get(col, "")).strip().lower()
            score = 1.0 if val in ["yes", "true", "1"] else 0.0
            utility += weights[col] * score

    if "price_bracket" in weights:
        price_map = {"2-3": 1.00, "3-4": 0.75, "4-5": 0.50, "5-6.5": 0.25, "6.5+": 0.00}
        score = price_map.get(str(pub_row.get("price_bracket", "NA")).strip(), 0.50)
        utility += weights["price_bracket"] * score

    if "vibe_score" in weights:
        val = pub_row.get("vibe_score", "NA")
        score = 0.50 if pd.isna(val) or str(val).upper() in ["NA", "NONE", "NAN"] else np.clip(float(val) / 10.0, 0.0, 1.0)
        utility += weights["vibe_score"] * score

    if "footprint_sqm" in weights:
        val = pub_row.get("footprint_sqm", "NA")
        score = 0.50 if pd.isna(val) or str(val).upper() in ["NA", "NONE", "NAN"] else np.clip(float(val) / 500.0, 0.0, 1.0)
        utility += weights["footprint_sqm"] * score

    return utility

def generate_optimal_pub_crawl(start_lat_lon, 
                               end_lat_lon, 
                               num_pubs, 
                               ranked_weights, 
                               num_routes_to_sample=25, 
                               max_leg_meters=1600.0):
    
    start_pt = gpd.GeoSeries([Point(start_lat_lon[1], start_lat_lon[0])], crs=4326).to_crs(27700).iloc[0]
    end_pt = gpd.GeoSeries([Point(end_lat_lon[1], end_lat_lon[0])], crs=4326).to_crs(27700).iloc[0]

    start_node = ox.distance.nearest_nodes(G, X=start_pt.x, Y=start_pt.y)
    end_node = ox.distance.nearest_nodes(G, X=end_pt.x, Y=end_pt.y)

    route_nodes = ox.shortest_path(G, start_node, end_node, weight="length")
    route_edges = ox.routing.route_to_gdf(G, route_nodes, weight="length")
    path_line = linemerge(route_edges.geometry.tolist())

    total_length = path_line.length
    step_distance = total_length / (num_pubs - 1) if num_pubs > 1 else 0

    stop_points = []
    for i in range(num_pubs):
        dist = i * step_distance
        stop_points.append({"stop_sequence": i + 1, "distance_m": round(dist, 2), "geometry": path_line.interpolate(dist)})
    interpolated_points = gpd.GeoDataFrame(stop_points, crs="EPSG:27700")

    interpolated_points["dist_along_line"] = interpolated_points.geometry.apply(lambda p: path_line.project(p))
    interpolated_points = interpolated_points.sort_values("dist_along_line").reset_index(drop=True)

    pubs = PUBS_GDF.copy()
    pub_id_col = "osmid" if "osmid" in pubs.columns else ("pub_name" if "pub_name" in pubs.columns else pubs.index.name)
    used_pub_ids = set()
    matched_records = []
    total_pts = len(interpolated_points)

    for idx, point in interpolated_points.iterrows():
        progress = idx / total_pts
        weight = 50.0 if progress < 0.50 else (30.0 if progress < 0.80 else 0.0)

        avail = pubs[~pubs[pub_id_col].isin(used_pub_ids)].copy() if pub_id_col else pubs.copy()
        if avail.empty:
            break

        avail["raw_dist_m"] = avail.geometry.distance(point.geometry)
        valid = avail[avail["raw_dist_m"] <= 600.0].copy()
        if valid.empty:
            valid = avail[avail["raw_dist_m"] <= 900.0].copy()
        if valid.empty:
            continue

        valid["effective_dist_m"] = (valid["raw_dist_m"] - (valid["density_score"] * weight)).clip(lower=0)
        selected = valid.sort_values("effective_dist_m").head(3)

        for _, pub in selected.iterrows():
            if pub_id_col:
                used_pub_ids.add(pub[pub_id_col])
            row = point.to_dict()
            row["geometry"] = pub.geometry
            row["snap_distance_m"] = round(pub["raw_dist_m"], 2)
            row["effective_dist_m"] = round(pub["effective_dist_m"], 2)
            for col in pubs.columns:
                if col != "geometry":
                    row[col] = pub[col]
            matched_records.append(row)

    candidates = gpd.GeoDataFrame(matched_records, crs="EPSG:27700")
    candidates = candidates.sort_values("dist_along_line").reset_index(drop=True)
    unique_points = sorted(candidates["dist_along_line"].unique())
    point_groups = [candidates[candidates["dist_along_line"] == p_dist] for p_dist in unique_points]

    routes_list = []
    for route_id in range(1, num_routes_to_sample + 1):
        raw_selected_pubs = []
        used = set()

        start_pub = point_groups[0].sample(1).iloc[0]
        raw_selected_pubs.append(start_pub.to_dict())
        used.add(start_pub[pub_id_col])
        last_idx = 0

        for _ in range(num_pubs - 2):
            max_idx = len(point_groups) - 2
            next_idx = random.randint(last_idx, min(last_idx + 2, max_idx)) if last_idx < max_idx else max_idx
            f_pool = point_groups[next_idx]
            avail = f_pool[~f_pool[pub_id_col].isin(used)]

            while avail.empty and next_idx < max_idx:
                next_idx += 1
                f_pool = point_groups[next_idx]
                avail = f_pool[~f_pool[pub_id_col].isin(used)]

            if avail.empty:
                break
            nxt = avail.sample(1).iloc[0]
            raw_selected_pubs.append(nxt.to_dict())
            used.add(nxt[pub_id_col])
            last_idx = next_idx

        end_pool = point_groups[-1]
        avail_end = end_pool[~end_pool[pub_id_col].isin(used)]
        end_pub = avail_end.sample(1).iloc[0] if not avail_end.empty else end_pool.sample(1).iloc[0]
        raw_selected_pubs.append(end_pub.to_dict())

        sample_gdf = gpd.GeoDataFrame(raw_selected_pubs, crs="EPSG:27700")
        sample_gdf["path_progress"] = sample_gdf["geometry"].apply(lambda geom: path_line.project(geom))
        sample_gdf = sample_gdf.sort_values("path_progress").reset_index(drop=True)
        sample_gdf["stop_sequence"] = range(1, len(sample_gdf) + 1)

        network_distances = [0.0]
        path_segments = []

        for i in range(1, len(sample_gdf)):
            p_geom, c_geom = sample_gdf.iloc[i - 1]["geometry"], sample_gdf.iloc[i]["geometry"]
            o_node = ox.distance.nearest_nodes(G, X=p_geom.x, Y=p_geom.y)
            d_node = ox.distance.nearest_nodes(G, X=c_geom.x, Y=c_geom.y)

            try:
                node_path = nx.shortest_path(G, o_node, d_node, weight="length")
                net_dist = nx.shortest_path_length(G, o_node, d_node, weight="length")
                line_coords = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in node_path]
                path_segments.append(LineString(line_coords) if len(line_coords) >= 2 else LineString([p_geom.coords[0], c_geom.coords[0]]))
            except Exception:
                net_dist = p_geom.distance(c_geom)
                path_segments.append(LineString([p_geom.coords[0], c_geom.coords[0]]))

            network_distances.append(round(net_dist, 2))

        sample_gdf["network_dist_from_prev_m"] = network_distances

        if sample_gdf["network_dist_from_prev_m"].max() <= max_leg_meters and len(sample_gdf) == num_pubs:
            line_gdf = gpd.GeoDataFrame([{"geometry": LineString([pt for seg in path_segments for pt in seg.coords])}], crs="EPSG:27700") if path_segments else None
            routes_list.append({"pubs": sample_gdf, "path": line_gdf})

    scored_routes = []
    for route in routes_list:
        pubs_gdf = route["pubs"]
        total_gravity = 0.0
        for _, pub in pubs_gdf.iterrows():
            mass = calculate_pub_utility(pub, ranked_weights)
            dist_m = pub["network_dist_from_prev_m"]
            gravity = mass if (pub.get("stop_sequence", 0) == 1 or dist_m == 0) else mass / max(dist_m, 10.0)
            total_gravity += gravity
        avg_gravity = total_gravity / len(pubs_gdf)
        scored_routes.append({"gravity_score": avg_gravity, "pubs": pubs_gdf, "path": route["path"]})

    return sorted(scored_routes, key=lambda x: x["gravity_score"], reverse=True)

# ---------------------------------------------------------
# 3. STREAMLIT INTERACTIVE UI & SESSION STATE
# ---------------------------------------------------------
PARAM_DISPLAY_MAP = {
    "Low Prices": "price_bracket",
    "Pool Table": "pool_table",
    "Beer Garden": "beer_garden",
    "Atmosphere": "vibe_score",
    "Cocktails": "cocktails",
    "More Floor Space": "footprint_sqm",
}

# Persistent State Setup
if "start_coords" not in st.session_state:
    st.session_state.start_coords = None
if "end_coords" not in st.session_state:
    st.session_state.end_coords = None
if "route_result" not in st.session_state:
    st.session_state.route_result = None

st.sidebar.header("Plan Your Bristol Pub Crawl! 🍺🍺🍺 ")
st.sidebar.write("📍 **1. Click twice on the map** to set Start (Green) & End (Red) pins.")

if st.sidebar.button("Reset Map & Pins"):
    st.session_state.start_coords = None
    st.session_state.end_coords = None
    st.session_state.route_result = None
    st.rerun()

if st.session_state.start_coords:
    st.sidebar.success(f"Start: {st.session_state.start_coords[0]:.4f}, {st.session_state.start_coords[1]:.4f}")
if st.session_state.end_coords:
    st.sidebar.error(f"End: {st.session_state.end_coords[0]:.4f}, {st.session_state.end_coords[1]:.4f}")

num_pubs = st.sidebar.slider("2. How many pubs can you handle?", min_value=2, max_value=30, value=5)

selected_priorities = st.sidebar.multiselect(
    "3. Clear preferences and re-rank in the order of importance:",
    options=list(PARAM_DISPLAY_MAP.keys()),
    default=list(PARAM_DISPLAY_MAP.keys())
)

run_button = st.sidebar.button("GENERATE", use_container_width=True)

# ---------------------------------------------------------
# 4. ROUTE CALCULATION LOGIC
# ---------------------------------------------------------
if run_button:
    if not st.session_state.start_coords or not st.session_state.end_coords:
        st.error("⚠️ Please click twice on the map to set Start & End points first!")
    else:
        final_order = [PARAM_DISPLAY_MAP[p] for p in selected_priorities]
        alpha = 0.7
        raw_weights = [np.exp(-alpha * i) for i in range(len(final_order))]
        sum_weights = sum(raw_weights) if sum(raw_weights) > 0 else 1.0
        weights = {param: round(raw_weights[i] / sum_weights, 4) for i, param in enumerate(final_order)}

        with st.spinner("⚡ Calculating route..."):
            results = generate_optimal_pub_crawl(
                start_lat_lon=st.session_state.start_coords,
                end_lat_lon=st.session_state.end_coords,
                num_pubs=num_pubs,
                ranked_weights=weights,
            )

        if not results:
            st.error("❌ No valid routes found within leg distance limits.")
            st.session_state.route_result = None
        else:
            st.session_state.route_result = results[0]

# ---------------------------------------------------------
# 5. MAP RENDERING
# ---------------------------------------------------------
pubs_4326 = PUBS_GDF.to_crs(4326)
center_lat = pubs_4326.geometry.y.mean()
center_lon = pubs_4326.geometry.x.mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=14,
    zoom_control = False,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri"
)

# Render Start Pin (Green Cross)
if st.session_state.start_coords:
    start_cross_html = """
    <div style="
        font-family: sans-serif;
        font-weight: 900;
        font-size: 16px;
        color: #2A9D8F;
        text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;
        line-height: 1;
        text-align: center;
    ">✕</div>
    """
    folium.Marker(
        location=st.session_state.start_coords,
        icon=folium.DivIcon(html=start_cross_html, icon_size=(16, 16), icon_anchor=(8, 8)),
        popup="Start"
    ).add_to(m)

# Render End Pin (Red Cross)
if st.session_state.end_coords:
    end_cross_html = """
    <div style="
        font-family: sans-serif;
        font-weight: 900;
        font-size: 16px;
        color: #E63946;
        text-shadow: -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;
        line-height: 1;
        text-align: center;
    ">✕</div>
    """
    folium.Marker(
        location=st.session_state.end_coords,
        icon=folium.DivIcon(html=end_cross_html, icon_size=(16, 16), icon_anchor=(8, 8)),
        popup="End"
    ).add_to(m)

# Render Persisted Route
if st.session_state.route_result:
    top_route = st.session_state.route_result

    if top_route["path"] is not None:
        path_4326 = top_route["path"].to_crs(4326)
        line_geom = path_4326.geometry.iloc[0]
        lat_lon_coords = [(pt[1], pt[0]) for pt in line_geom.coords]
        folium.PolyLine(lat_lon_coords, color="#3A0CA3", weight=4, opacity=0.8).add_to(m)

    route_pubs_4326 = top_route["pubs"].to_crs(4326)
    for stop_idx, row in route_pubs_4326.iterrows():
        stop_num = stop_idx + 1
        lat, lon = row.geometry.y, row.geometry.x

        badge_html = f"""
        <div style="
            background-color: #1D3557; 
            color: white; 
            border-radius: 50%; 
            width: 18px; 
            height: 18px; 
            display: flex; 
            align-items: center; 
            justify-content: center; 
            font-weight: bold; 
            font-size: 10px; 
            font-family: sans-serif;
            box-shadow: 0 1px 3px rgba(0,0,0,0.4);
            border: 1.5px solid white;
        ">{stop_num}</div>
        """

        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=badge_html, icon_size=(18, 18), icon_anchor=(9, 9)),
            popup=f"<b>Stop {stop_num}: {row['pub_name']}</b>",
            tooltip=f"Stop {stop_num}: {row['pub_name']}"
        ).add_to(m)

# Render Map Component
map_data = st_folium(
    m, 
    use_container_width=True,  # Automatically scales to device screen width
    height=420,                 # Fits nicely on mobile screens without pushing UI off
    key="pub_map",
    returned_objects=["last_clicked"]  # Prevents unnecessary heavy re-renders
)

# Display Sequence Below Map
if st.session_state.route_result:
    route_pubs_4326 = st.session_state.route_result["pubs"].to_crs(4326)
    st.subheader("📋 Stop Sequence")
    for idx, row in route_pubs_4326.iterrows():
        st.write(f"**Stop {idx+1}:** {row['pub_name']}")

# Capture Click Events
if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lng = map_data["last_clicked"]["lng"]
    new_point = (clicked_lat, clicked_lng)

    if st.session_state.start_coords is None:
        st.session_state.start_coords = new_point
        st.session_state.route_result = None
        st.rerun()
    elif st.session_state.end_coords is None and new_point != st.session_state.start_coords:
        st.session_state.end_coords = new_point
        st.session_state.route_result = None
        st.rerun()