import os
import re
import json
import requests
import shutil
import threading
import time
from collections import defaultdict
from pyproj import Transformer
from shapely.geometry import shape

# Define base folders.
BASE_FOLDER = "Geojson"         # Fetched data and original GeoJSON files.
TILES_FOLDER_BASE = "Tiles"     # Final outputs including partitions and copied json/centroid files.
os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(TILES_FOLDER_BASE, exist_ok=True)

# Global transformer: EPSG:3857 (Web Mercator) to EPSG:4326 (WGS84)
TO_WGS84 = Transformer.from_crs("epsg:3857", "epsg:4326", always_xy=True)

# --- Helper: Timed input using threading (works cross-platform) ---
def timed_input(prompt, timeout=5):
    """
    Wait for user input for a limited time (default 5 seconds).
    If no input is provided within the timeout, return an empty string.
    """
    print(prompt, end="", flush=True)
    user_input = []
    def input_worker():
        try:
            user_input.append(input())
        except Exception:
            pass
    t = threading.Thread(target=input_worker)
    t.daemon = True
    t.start()
    t.join(timeout)
    if user_input:
        return user_input[0]
    else:
        print(f"\nNo input provided in {timeout} seconds. Proceeding with default (no renaming).")
        return ""

def sanitize_filename(filename):
    """
    Replace characters that are not allowed in file names with a blank space.
    On Windows, characters like <>:"/\\|?*+ are invalid.
    """
    return re.sub(r'[<>:"/\\|?*+]', ' ', filename)

def sanitize_base_url(base_url):
    """Extract the base URL up to /query."""
    m = re.match(r'^(https?://[^?]+/query)', base_url)
    return m.group(1) if m else base_url

def get_server_max_record_count(base_url):
    """
    Query the ArcGIS service metadata to obtain the server's maxRecordCount.
    If the request fails or the property is missing, default to 1000.
    """
    params = {"f": "json"}
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        max_count = data.get("maxRecordCount", 2000)
        print(f"Server maxRecordCount is {max_count}.")
        return max_count
    except requests.RequestException as e:
        print(f"Error retrieving maxRecordCount from the server: {e}")
        return 1000

def fetch_features_by_tehsil(base_url, tehsil_name):
    """
    Fetch all features from the ArcGIS service where the 'Tehsil' attribute equals tehsil_name.
    Uses pagination and, if an error occurs, retries the same offset until a successful response is obtained or no features remain.
    """
    max_record_count = get_server_max_record_count(base_url)
    features = []
    offset = 0
    while True:
        params = {
            "where": f"Tehsil='{tehsil_name}'",
            "outFields": "*",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": max_record_count
        }
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching features for Tehsil {tehsil_name} at offset {offset}: {e}")
            time.sleep(2)
            continue

        if "features" not in data:
            print("Response does not contain 'features'. Retrying...")
            time.sleep(2)
            continue

        if not data["features"]:
            break

        features.extend(data["features"])
        print(f"Fetched {len(data['features'])} features at offset {offset}.")
        if len(data["features"]) < max_record_count:
            break

        offset += max_record_count

    print(f"Total features fetched for Tehsil '{tehsil_name}': {len(features)}")
    return features

def reproject_geometry(geometry):
    """
    Transform ArcGIS geometry to GeoJSON format, converting coordinates from EPSG:3857 to EPSG:4326.
    Supports Polygon, MultiLineString, and Point geometries.
    """
    if "rings" in geometry:
        transformed_rings = []
        for ring in geometry["rings"]:
            transformed_ring = []
            for x, y in ring:
                lon, lat = TO_WGS84.transform(x, y)
                transformed_ring.append([lon, lat])
            transformed_rings.append(transformed_ring)
        return {"type": "Polygon", "coordinates": transformed_rings}
    elif "paths" in geometry:
        transformed_paths = []
        for path in geometry["paths"]:
            transformed_path = []
            for x, y in path:
                lon, lat = TO_WGS84.transform(x, y)
                transformed_path.append([lon, lat])
            transformed_paths.append(transformed_path)
        return {"type": "MultiLineString", "coordinates": transformed_paths}
    elif "x" in geometry and "y" in geometry:
        lon, lat = TO_WGS84.transform(geometry["x"], geometry["y"])
        return {"type": "Point", "coordinates": [lon, lat]}
    return None

def create_killa(properties):
    """
    Create the Killa field based on attributes K, A, and SK.
    Modified logic:
      - Retrieve K from the properties (defaulting to "Unknown").
      - If SK is empty:
           * If K equals A then return A.
           * Otherwise return K.
      - If SK is non-empty:
           * Use A if it is non-empty, otherwise use K.
           * Concatenate the chosen value with SK using an underscore.
    """
    k = properties.get("K", "Unknown")
    a = str(properties.get("A") or "").strip()
    sk = str(properties.get("SK") or "").strip()

    if not sk:
        if k == a:
            return a
        else:
            return k
    else:
        chosen = a if a else k
        return f"{chosen}_{sk}"

def convert_to_geojson_feature(feature):
    """
    Convert an ArcGIS feature (with 'attributes' and 'geometry') to a GeoJSON feature.
    Also extracts key attributes: murabba_no, tehsil_name, and mouza_value.
    """
    geometry = feature.get("geometry")
    properties = feature.get("attributes", {})
    # Compute and add the Killa attribute.
    properties["Killa"] = create_killa(properties)
    murabba_no = properties.get("M", "Unknown")
    tehsil_name = properties.get("Tehsil", "Unknown_Tehsil")
    mouza_value = properties.get("Mouza", "Unknown_Mouza")
    if geometry is None:
        return None, murabba_no, tehsil_name, properties["Killa"], mouza_value
    geojson_geometry = reproject_geometry(geometry)
    geojson_feature = {
        "type": "Feature",
        "geometry": geojson_geometry,
        "properties": properties
    }
    return geojson_feature, murabba_no, tehsil_name, properties["Killa"], mouza_value

def compute_bounds(features):
    """
    Compute the bounding box for a list of GeoJSON features.
    Returns a dictionary with the four corner coordinates.
    """
    min_lon = float("inf")
    min_lat = float("inf")
    max_lon = float("-inf")
    max_lat = float("-inf")
    for feature in features:
        geom = feature.get("geometry")
        if not geom:
            continue
        coords = []
        if geom["type"] == "Point":
            coords = [geom["coordinates"]]
        elif geom["type"] == "Polygon":
            for ring in geom["coordinates"]:
                coords.extend(ring)
        elif geom["type"] == "MultiLineString":
            for line in geom["coordinates"]:
                coords.extend(line)
        for lon, lat in coords:
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
    if min_lon == float("inf") or min_lat == float("inf"):
        return None
    return {
        "topLeft": [min_lon, max_lat],
        "topRight": [max_lon, max_lat],
        "bottomRight": [max_lon, min_lat],
        "bottomLeft": [min_lon, min_lat]
    }

def create_bounds_json(mouza_name, features, output_file):
    """
    Create a JSON file containing the bounding box of the given features.
    """
    bounds = compute_bounds(features)
    if bounds:
        xmin = bounds["bottomLeft"][0]
        ymin = bounds["bottomLeft"][1]
        xmax = bounds["topRight"][0]
        ymax = bounds["topRight"][1]
        bounds_str = f"{xmin},{ymin},{xmax},{ymax}"
        bounds_data = {
            "name": mouza_name,
            "format": "png",
            "minZoom": 10,
            "maxZoom": 19,
            "bounds": bounds_str
        }
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(bounds_data, f, indent=2)
            print(f"Mouza bounds saved to {output_file}")
        except IOError as e:
            print(f"Error writing bounds file {output_file}: {e}")
    else:
        print("Could not compute bounds for the features.")

def create_centroid_features_by_m(features):
    """
    For a list of polygon features, group them by their 'M' attribute and compute a single
    centroid point for each group, where the centroid is the average of individual polygon centroids.
    Returns a list of GeoJSON point features.
    """
    groups = defaultdict(list)
    for feature in features:
        if feature.get("geometry") and feature["geometry"]["type"] == "Polygon":
            m_value = feature.get("properties", {}).get("M", "Unknown")
            groups[m_value].append(feature)

    centroid_features = []
    for m_value, feat_list in groups.items():
        sum_x = 0.0
        sum_y = 0.0
        count = 0
        for feat in feat_list:
            polygon = shape(feat["geometry"])
            centroid = polygon.centroid
            sum_x += centroid.x
            sum_y += centroid.y
            count += 1
        if count > 0:
            avg_x = sum_x / count
            avg_y = sum_y / count
            centroid_geojson = {"type": "Point", "coordinates": [avg_x, avg_y]}
            properties = {"Murabba_No": m_value, "Label": m_value}
            centroid_features.append({
                "type": "Feature",
                "geometry": centroid_geojson,
                "properties": properties
            })
    return centroid_features

def main():
    base_url = input("Enter Base URL (up to /query): ").strip()
    base_url = sanitize_base_url(base_url)

    tehsil_name_input = input("Enter Tehsil name: ").strip()
    if not tehsil_name_input:
        print("Tehsil name not provided; exiting.")
        return

    # Create the tehsil folder inside the Geojson base folder.
    sanitized_tehsil = sanitize_filename(tehsil_name_input.replace(" ", "_"))
    tehsil_folder = os.path.join(BASE_FOLDER, sanitized_tehsil)
    os.makedirs(tehsil_folder, exist_ok=True)

    # Fetch all features for the tehsil.
    raw_features = fetch_features_by_tehsil(base_url, tehsil_name_input)
    if not raw_features:
        print("No features were retrieved; exiting.")
        return

    # Convert features to GeoJSON and group them by the 'Mouza' attribute.
    grouped_features = {}
    for feat in raw_features:
        converted = convert_to_geojson_feature(feat)
        if not converted[0]:
            continue
        geojson_feature, murabba_no, tehsil_name, killa_value, mouza_value = converted
        if mouza_value not in grouped_features:
            grouped_features[mouza_value] = []
        grouped_features[mouza_value].append(geojson_feature)

    # Create subfolders for bounds JSON and centroid GeoJSON files inside tehsil_folder.
    json_folder = os.path.join(tehsil_folder, "json")
    centroid_folder = os.path.join(tehsil_folder, "centroid")
    os.makedirs(json_folder, exist_ok=True)
    os.makedirs(centroid_folder, exist_ok=True)

    # Dictionary to store the current file name for each mouza.
    mouza_names = {}
    # Process each mouza group.
    for mouza, features in grouped_features.items():
        safe_mouza = sanitize_filename(mouza)
        mouza_names[mouza] = safe_mouza

        # Swap the values of Label and Killa for each feature.
        for feature in features:
            props = feature.get("properties", {})
            killa_val = props.get("Killa")
            if not isinstance(killa_val, str):
                killa_val = str(killa_val)
            current_label = props.get("Label", "")
            props["Label"] = killa_val
            props["Killa"] = current_label

        # Save the original features GeoJSON file in tehsil_folder.
        feature_file = os.path.join(tehsil_folder, f"{safe_mouza}.geojson")
        with open(feature_file, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f, indent=2)
        print(f"Saved {len(features)} features for Mouza '{mouza}' to {feature_file}")

        # Create and save the bounds JSON file.
        bounds_file = os.path.join(json_folder, f"{safe_mouza}.json")
        create_bounds_json(mouza, features, bounds_file)

        # Create and save the centroid GeoJSON file.
        centroid_features = create_centroid_features_by_m(features)
        if centroid_features:
            centroid_file = os.path.join(centroid_folder, f"{safe_mouza}.geojson")
            with open(centroid_file, "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": centroid_features}, f, indent=2)
            print(f"Saved {len(centroid_features)} centroid features for Mouza '{mouza}' to {centroid_file}")
        else:
            print(f"No valid centroid features for Mouza '{mouza}'.")

    # Instead of zipping, copy all json and centroid files to Tiles/[Tehsil].
    tiles_folder = os.path.join(TILES_FOLDER_BASE, sanitized_tehsil)
    os.makedirs(tiles_folder, exist_ok=True)
    for folder in [json_folder, centroid_folder]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    dest_path = os.path.join(tiles_folder, file)
                    try:
                        shutil.copy(file_path, dest_path)
                        print(f"Copied {file_path} to {dest_path}")
                    except Exception as e:
                        print(f"Error copying {file_path} to {dest_path}: {e}")

    # Clean up: remove the json and centroid subfolders from the Geojson tehsil folder.
    for folder in [json_folder, centroid_folder]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
            try:
                os.rmdir(folder)
            except Exception as e:
                print(f"Error deleting folder {folder}: {e}")
    print("Cleaned up individual bounds and centroid files from Geojson folder.")

    # --- Renaming Prompt ---
    rename_choice = timed_input("Do you want to rename the mauza names? (Y/N): ", timeout=5).strip().lower()
    if rename_choice.startswith('y'):
        for original_mouza, current_name in mouza_names.items():
            current_file = os.path.join(tehsil_folder, f"{current_name}.geojson")
            new_name_input = input(f"Enter new name for mouza '{current_name}' (or press Enter to keep unchanged): ").strip()
            if new_name_input:
                new_safe_name = sanitize_filename(new_name_input)
                new_file = os.path.join(tehsil_folder, f"{new_safe_name}.geojson")
                try:
                    os.rename(current_file, new_file)
                    mouza_names[original_mouza] = new_safe_name
                    print(f"Renamed '{current_name}' to '{new_safe_name}'.")
                except Exception as e:
                    print(f"Error renaming file {current_file} to {new_file}: {e}")
    else:
        print("Keeping all mauza names as is.")

    # --- Partitioning: Copy original GeoJSON files into 3 subfolders in Tiles/[Tehsil] ---
    part_folder_names = [f"{sanitized_tehsil}-1", f"{sanitized_tehsil}-2", f"{sanitized_tehsil}-3"]
    part_folders = []
    for folder_name in part_folder_names:
        folder_path = os.path.join(tiles_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        part_folders.append(folder_path)

    # Collect all original GeoJSON files from the Geojson tehsil folder.
    geojson_files = []
    for item in os.listdir(tehsil_folder):
        item_path = os.path.join(tehsil_folder, item)
        if os.path.isfile(item_path) and item.lower().endswith(".geojson"):
            geojson_files.append(item_path)

    # Greedy algorithm to distribute copies by file size.
    bucket_sizes = [0, 0, 0]
    bucket_files = {0: [], 1: [], 2: []}
    for file_path in geojson_files:
        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0
        bucket_index = bucket_sizes.index(min(bucket_sizes))
        bucket_files[bucket_index].append(file_path)
        bucket_sizes[bucket_index] += size

    # Copy files to the corresponding partition folders in Tiles.
    for bucket_index, files in bucket_files.items():
        for file_path in files:
            file_name = os.path.basename(file_path)
            destination = os.path.join(part_folders[bucket_index], file_name)
            try:
                shutil.copy(file_path, destination)
            except Exception as e:
                print(f"Error copying file {file_path} to {destination}: {e}")
    print("Copied mauza GeoJSON files into 3 partition folders in Tiles folder for balanced sizes.")

    # Create a tehsil text file listing all processed (and possibly renamed) mauza file names in Tiles folder.
    final_names = list(mouza_names.values())
    tehsil_text_file = os.path.join(tiles_folder, f"{sanitized_tehsil}.txt")
    try:
        with open(tehsil_text_file, "w", encoding="utf-8") as f:
            f.write(",".join(final_names))
        print(f"List of mauza file names saved to {tehsil_text_file}")
    except IOError as e:
        print(f"Error writing tehsil text file {tehsil_text_file}: {e}")

if __name__ == "__main__":
    main()
