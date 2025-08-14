import re
import requests
import json
import os
import time
import random
import logging
import shutil
from itertools import cycle
from pyproj import Transformer
from tqdm import tqdm
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
from tkinter import filedialog

# -- CONFIGURATION --
OUTPUT_DIR = "Output"  # kept for compatibility if user cancels dialog
TOKEN_FILE = "token.txt"
PROXY_FILE = "PROXYLIST.txt"
DISTRICT_FILE = "district.json"
LOG_FILE = "download_tehsil.log"
MAX_WORKERS = 2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

TO_WGS84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

def sanitize_filename(name):
    # Keep spaces as spaces, replace special characters with dash
    return re.sub(r'[\\/:*?"<>|]', '-', str(name).strip())

def ask_output_folder(title="Select destination folder for output"):
    """Ask user to pick a destination root folder via dialog. Return None if cancelled."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title, mustexist=True)
    root.destroy()
    return folder or None

def load_tokens():
    with open(TOKEN_FILE, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]
    if not tokens:
        raise Exception("No tokens found in token.txt")
    while True:
        for token in tokens:
            yield token

def parse_proxy(line):
    parts = line.strip().split(':')
    if len(parts) == 4:
        ip, port, user, pwd = parts
        return f"http://{user}:{pwd}@{ip}:{port}"
    elif len(parts) == 2:
        ip, port = parts
        return f"http://{ip}:{port}"
    raise ValueError(f"Invalid proxy format: {line}")

def load_proxies():
    with open(PROXY_FILE, "r") as f:
        proxies = [parse_proxy(line) for line in f if line.strip()]
    if not proxies:
        raise Exception("No proxies found in PROXYLIST.txt")
    while True:
        for proxy in proxies:
            yield proxy

def fetch_json(url, proxy):
    headers = {
        "User-Agent": f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(115, 125)}.0.0.0 Safari/537.36"
    }
    proxies = {"http": proxy, "https": proxy}
    for attempt in range(3):
        try:
            r = requests.get(url, headers=headers, timeout=30, proxies=proxies)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logging.warning(f"Request failed (attempt {attempt+1}) [{proxy}] {url}: {e}")
            time.sleep(2 + attempt)
    raise RuntimeError(f"Failed after retries on proxy: {proxy}")

def fetch_tehsil_list(base_url, token, proxy):
    url = f"{base_url}/query?where=1=1&outFields=Tehsil_ID,Tehsil&returnDistinctValues=true&returnGeometry=false&f=json&token={token}"
    resp = fetch_json(url, proxy)
    return [{"id": f["attributes"]["Tehsil_ID"], "name": f["attributes"]["Tehsil"]} for f in resp.get("features", [])]

def get_total_feature_count(base_url, token, tehsil_id, proxy):
    url = f"{base_url}/query?where=Tehsil_ID={tehsil_id}&returnCountOnly=true&f=json&token={token}"
    resp = fetch_json(url, proxy)
    return resp.get("count", 0)

def fetch_tehsil_batch(base_url, token, tehsil_id, offset, batch_size, proxy):
    fields = "Tehsil,Tehsil_ID,Mouza,Mouza_ID,Type,M,A,K,SK,Label,B,MN"
    url = (f"{base_url}/query?where=Tehsil_ID={tehsil_id}&outFields={fields}&returnGeometry=true&f=json"
           f"&resultOffset={offset}&resultRecordCount={batch_size}&token={token}")
    resp = fetch_json(url, proxy)
    return resp.get("features", [])

def reproject_polygon(rings):
    return [
        [
            list(TO_WGS84.transform(x, y))
            for x, y in ring
        ]
        for ring in rings
    ]

def compute_bounds(features):
    min_lon, min_lat = float("inf"), float("inf")
    max_lon, max_lat = float("-inf"), float("-inf")
    for feat in features:
        geom = feat.get("geometry")
        if not geom:
            continue
        coords = []
        if geom["type"] == "Polygon":
            for ring in geom["coordinates"]:
                coords.extend(ring)
        elif geom["type"] == "Point":
            coords = [geom["coordinates"]]
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
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(bounds_data, f, indent=2)

def get_group_key(props):
    typ = str(props.get("Type", "")).strip()
    m_val = props.get("M")
    mn_val = props.get("MN")
    b_val = props.get("B")
    k_val = props.get("K")
    a_val = props.get("A")

    # Normalize values for blank/zero checks
    def norm(x):
        return None if x in [None, "", "None"] else (int(x) if str(x).isdigit() else x)

    m_norm = norm(m_val)
    mn_norm = norm(mn_val)
    b_norm = norm(b_val)
    k_norm = norm(k_val)
    a_norm = norm(a_val)

    # --- KW and MU/KW special logic
    if typ in ("KW", "MU/KW"):
        if m_norm not in [None, 0]:
            return str(m_norm)
        if a_norm not in [None, 0]:
            return str(a_norm)
        if k_norm not in [None, 0]:
            return str(k_norm)
        return "0"
    # --- Existing logic for MT
    if typ == "MT":
        # Normalize condition flags
        m_valid = m_norm not in [None, 0]
        mn_valid = mn_norm not in [None, 0]
        b_valid = b_norm not in [None, "", "None", 0, "0"]

        # Rule 1: All missing or zero
        if not m_valid and not mn_valid:
            return "0"

        # Rule 2: All three are non-zero → use M
        if m_valid and mn_valid and b_valid:
            return str(m_norm)

        # Rule 3: M is zero or missing, MN and B exist → return B/MN
        if not m_valid and mn_valid and b_valid:
            return f"{b_norm}/{mn_norm}"

        # Rule 4: Only M is valid
        if m_valid and not mn_valid:
            return str(m_norm)

        # Rule 5: Only MN is valid
        if mn_valid and not m_valid:
            return str(mn_norm)

        # Rule 6: M and MN exist and are equal → return M
        if m_valid and mn_valid and m_norm == mn_norm:
            return str(m_norm)

        # Rule 7: fallback to MN if valid
        if mn_valid:
            return str(mn_norm)

        # Rule 8: fallback to M if valid
        if m_valid:
            return str(m_norm)

        return "0"

    # --- Existing logic for MU, MT/MU, MU/MT
    elif typ in ("MU", "MU/MT", "MT/MU"):
        if m_norm not in [None, 0]:
            return str(m_norm)
    elif typ == "K":
        if k_norm not in [None, 0]:
            return str(k_norm)
    # Fallback: use M if present
    if m_norm not in [None, 0]:
        return str(m_norm)
    return "0"

def create_grouped_centroids(features):
    from shapely.ops import unary_union
    from collections import defaultdict

    grouped = defaultdict(list)
    for feat in features:
        props = feat.get("properties", {})
        key = get_group_key(props)
        if key is None or key == "0":
            continue
        geom = feat.get("geometry")
        if not geom or geom["type"] != "Polygon":
            continue
        poly = shapely_shape(geom)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_valid and not poly.is_empty:
            grouped[key].append(poly)
        else:
            print(f"Warning: Skipped invalid geometry for Murabba_No={key}")
    centroid_features = []
    for murabba_no, geom_list in grouped.items():
        if not geom_list:
            continue
        try:
            union_geom = unary_union(geom_list)
            centroid = union_geom.centroid
            centroid_feature = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [centroid.x, centroid.y]},
                "properties": {"Murabba_No": murabba_no}
            }
            centroid_features.append(centroid_feature)
        except Exception as e:
            print(f"Error in union/centroid for Murabba_No={murabba_no}: {e}")
    return centroid_features

def group_and_save(features, district_name, tehsil_name, custom_output_folder=None):
    grouped = {}
    for feat in features:
        props = feat["attributes"]
        mouza = sanitize_filename(props.get("Mouza", "Unknown"))
        geom = feat.get("geometry")
        if geom and "rings" in geom:
            wgs84_coords = reproject_polygon(geom["rings"])
            gj_feat = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": wgs84_coords},
                "properties": props
            }
            grouped.setdefault(mouza, []).append(gj_feat)

    if custom_output_folder:
        tehsil_folder = custom_output_folder
    else:
        tehsil_folder = os.path.join(OUTPUT_DIR, sanitize_filename(district_name), sanitize_filename(tehsil_name))
    os.makedirs(tehsil_folder, exist_ok=True)
    for mouza, feats in grouped.items():
        geojson_path = os.path.join(tehsil_folder, f"{mouza}.geojson")
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": feats}, f, indent=2)
        bounds_path = os.path.join(tehsil_folder, f"{mouza}.json")
        create_bounds_json(mouza, feats, bounds_path)
        centroid_feats = create_grouped_centroids(feats)
        if centroid_feats:
            centroid_path = os.path.join(tehsil_folder, f"{mouza}_centroid.geojson")
            with open(centroid_path, "w", encoding="utf-8") as f:
                json.dump({"type": "FeatureCollection", "features": centroid_feats}, f, indent=2)
            print(f"Saved polygons, {len(centroid_feats)} centroid(s), bounds for Mouza '{mouza}'.")
        else:
            print(f"No centroids found for Mouza '{mouza}'.")

def zip_and_cleanup_centroids_and_bounds(district_name, tehsil_name, custom_output_folder=None):
    if custom_output_folder:
        tehsil_folder = custom_output_folder
        zip_path = os.path.join(os.path.dirname(custom_output_folder), f"{sanitize_filename(tehsil_name)}.zip")
    else:
        tehsil_folder = os.path.join(OUTPUT_DIR, sanitize_filename(district_name), sanitize_filename(tehsil_name))
        zip_path = os.path.join(OUTPUT_DIR, sanitize_filename(district_name), f"{sanitize_filename(tehsil_name)}.zip")

    files_to_zip = []
    for fn in os.listdir(tehsil_folder):
        if fn.endswith('_centroid.geojson'):
            new_name = fn.replace('_centroid.geojson', '.geojson')
            files_to_zip.append((os.path.join(tehsil_folder, fn), new_name))
        elif fn.endswith('.json') and not fn.endswith('-raw.geojson'):
            files_to_zip.append((os.path.join(tehsil_folder, fn), fn))

    if not files_to_zip:
        print("No centroid or bounds files found to zip.")
        return

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in files_to_zip:
            zf.write(file_path, arcname=arcname)
    print(f"\nZipped {len(files_to_zip)} centroid/bounds files to {zip_path}")

    for file_path, _ in files_to_zip:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete {file_path}: {e}")
    print("Cleaned up zipped centroid and bounds files.\n")

def save_batch(tehsil_folder, offset, features):
    batch_dir = os.path.join(tehsil_folder, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    batch_path = os.path.join(batch_dir, f"batch_{offset}.json")
    with open(batch_path, "w", encoding="utf-8") as f:
        json.dump(features, f)
    return batch_path

def read_completed_offsets(tehsil_folder):
    progress_file = os.path.join(tehsil_folder, "download.progress")
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())

def mark_offset_done(tehsil_folder, offset):
    progress_file = os.path.join(tehsil_folder, "download.progress")
    with open(progress_file, "a") as f:
        f.write(f"{offset}\n")

def load_all_batches(tehsil_folder):
    batch_dir = os.path.join(tehsil_folder, "batches")
    all_feats = []
    if os.path.exists(batch_dir):
        for fname in os.listdir(batch_dir):
            if fname.startswith("batch_") and fname.endswith(".json"):
                with open(os.path.join(batch_dir, fname), "r", encoding="utf-8") as f:
                    feats = json.load(f)
                    all_feats.extend(feats)
    return all_feats

def ask_for_raw_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Raw GeoJSON File",
        filetypes=[("GeoJSON Files", "*.geojson"), ("All Files", "*.*")]
    )
    root.destroy()
    return file_path if file_path else None

def fetch_and_save_batch(args):
    tehsil_id, offset, batch_size, base_url, tokens, proxies, tehsil_folder = args
    token = next(tokens)
    proxy = next(proxies)
    try:
        feats = fetch_tehsil_batch(base_url, token, tehsil_id, offset, batch_size, proxy)
        save_batch(tehsil_folder, offset, feats)
        mark_offset_done(tehsil_folder, offset)
        msg = f"[OK] Batch offset {offset} (size {len(feats)}) done for Tehsil_ID={tehsil_id}"
    except Exception as e:
        msg = f"[FAILED] Batch offset {offset} for Tehsil_ID={tehsil_id}: {e}"
    time.sleep(random.uniform(1, 3))
    return msg

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- STEP 1: Ask about raw data globally ---
    answer = input("Is the raw data (GeoJSON) already downloaded? (y/n): ").strip().lower()
    if answer == 'y':
        raw_file = ask_for_raw_file()
        if not raw_file or not os.path.exists(raw_file):
            print("No file selected or file does not exist. Exiting.")
            exit(1)
        print(f"Selected raw file: {raw_file}")

        # Choose destination root for RAW branch
        dest_root = ask_output_folder("Select destination folder (root) for saving RAW output")
        if not dest_root:
            print("No destination folder selected. Exiting.")
            exit(1)

        with open(raw_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        unique_features = raw_data["features"]

        tehsil_name = None
        district_name = None
        if unique_features:
            attr = unique_features[0].get("attributes") or unique_features[0].get("properties")
            if attr:
                tehsil_name = attr.get("Tehsil") or attr.get("tehsil") or attr.get("tehsil_name")
                district_name = attr.get("District") or attr.get("district") or attr.get("district_name")
                if tehsil_name:
                    tehsil_name = sanitize_filename(str(tehsil_name))
                if district_name:
                    district_name = sanitize_filename(str(district_name))

        if not tehsil_name:
            tehsil_name = sanitize_filename(input("Enter Tehsil Name for output folder: ").strip())
        if not district_name:
            # District is optional here; if not provided, we won't create an extra level
            district_name = sanitize_filename(input("Enter District Name for output folder (optional, press Enter to skip): ").strip() or "")

        # Build output structure under chosen dest_root
        parent_folder = os.path.join(dest_root, district_name) if district_name else dest_root
        os.makedirs(parent_folder, exist_ok=True)
        tehsil_output_folder = os.path.join(parent_folder, tehsil_name)
        os.makedirs(tehsil_output_folder, exist_ok=True)

        # Only write raw if doesn't already exist
        raw_geojson_path = os.path.join(parent_folder, f"{tehsil_name}-raw.geojson")
        if not os.path.exists(raw_geojson_path):
            with open(raw_geojson_path, "w", encoding="utf-8") as f:
                json.dump(raw_data, f, indent=2)
            print(f"Raw GeoJSON written to: {raw_geojson_path}")
        else:
            print(f"Raw GeoJSON already exists at: {raw_geojson_path}, skipping write.")

        group_and_save(unique_features, None, tehsil_name, custom_output_folder=tehsil_output_folder)
        zip_and_cleanup_centroids_and_bounds(None, tehsil_name, custom_output_folder=tehsil_output_folder)
        print("Processing complete.")
        logging.info(f"Completed in {time.time() - t0:.1f}s")
        exit(0)

    elif answer != 'n':
        print("Invalid input. Please enter 'y' or 'n'.")
        exit(1)

    # --- REGULAR ARC-GIS DOWNLOAD WORKFLOW ---
    with open(DISTRICT_FILE, "r", encoding="utf-8") as f:
        districts = json.load(f)
    per_row = 3
    rows = (len(districts) + per_row - 1) // per_row

    print("\nAvailable Districts (ID | Name):")
    print("-" * 70)
    for row in range(rows):
        line = ""
        for col in range(per_row):
            idx = row + col * rows
            if idx < len(districts):
                d = districts[idx]
                entry = f"{d['id']:>2}: {d['name']:<20}"
                line += entry + "   "
        print(line)
    print("-" * 70)

    district_ids = [d["id"] for d in districts]
    while True:
        try:
            sel_id = int(input("\nEnter the District ID from above table: "))
            if sel_id in district_ids:
                sel_district = next(d for d in districts if d["id"] == sel_id)
                BASE_URL = sel_district["url"]
                district_name = sel_district["name"]
                break
            else:
                print("Invalid District ID, try again.")
        except Exception:
            print("Please enter a valid numeric ID.")

    # Ask once for destination folder for this district
    dest_root = ask_output_folder(f"Select destination folder (root) for saving district '{district_name}'")
    if not dest_root:
        print("No destination folder selected. Exiting.")
        exit(1)
    district_folder = os.path.join(dest_root, sanitize_filename(district_name))
    os.makedirs(district_folder, exist_ok=True)

    tokens = load_tokens()
    proxies = load_proxies()
    token = next(tokens)
    proxy = next(proxies)

    tehsils = fetch_tehsil_list(BASE_URL, token, proxy)
    print("\nAvailable Tehsils:")
    for t in tehsils:
        print(f"Tehsil: {t['name']} | ID: {t['id']}")

    tehsil_ids_input = input("\nTehsil ID(s) to process (comma separated): ").strip()
    tehsil_ids = [int(tid.strip()) for tid in tehsil_ids_input.split(",") if tid.strip().isdigit()]
    tehsil_map = {}
    for tid in tehsil_ids:
        matching = [t for t in tehsils if t['id'] == tid]
        if not matching:
            print(f"Warning: Tehsil ID {tid} not found in available tehsils.")
            continue
        tname = matching[0]['name']
        tehsil_map.setdefault(tname, []).append(tid)

    for tehsil_name, id_list in tehsil_map.items():
        tehsil_folder = os.path.join(district_folder, sanitize_filename(tehsil_name))
        os.makedirs(tehsil_folder, exist_ok=True)
        tehsil_raw_path = os.path.join(district_folder, f"{sanitize_filename(tehsil_name)}-raw.geojson")

        print(f"\nProcessing tehsil: {tehsil_name}")

        completed_offsets = read_completed_offsets(tehsil_folder)
        all_batches = []
        for tehsil_id in id_list:
            token = next(tokens)
            proxy = next(proxies)
            total_count = get_total_feature_count(BASE_URL, token, tehsil_id, proxy)
            print(f"Total features for Tehsil_ID={tehsil_id}: {total_count}")

            batch_sizes = []
            offsets = []
            current = 0
            while current < total_count:
                batch_size = random.randint(1000, 2000)
                batch_sizes.append(batch_size)
                offsets.append(current)
                current += batch_size
            all_batches.extend([(tehsil_id, offset, batch_sizes[i]) for i, offset in enumerate(offsets)])

        pending = [(tid, offset, bsize) for (tid, offset, bsize) in all_batches if offset not in completed_offsets]
        print(f"Pending batches for {tehsil_name}: {len(pending)}")

        max_retries = 3
        initial_work_args = [
            (tehsil_id, offset, batch_size, BASE_URL, tokens, proxies, tehsil_folder)
            for tehsil_id, offset, batch_size in pending
        ]

        # Track failed batches
        failed_batches = []

        def fetch_and_save_batch_with_status(args):
            """Returns (success_bool, message, original_args)"""
            tehsil_id, offset, batch_size, base_url, tokens, proxies, tehsil_folder = args
            token = next(tokens)
            proxy = next(proxies)
            try:
                feats = fetch_tehsil_batch(base_url, token, tehsil_id, offset, batch_size, proxy)
                save_batch(tehsil_folder, offset, feats)
                mark_offset_done(tehsil_folder, offset)
                msg = f"[OK] Batch offset {offset} (size {len(feats)}) done for Tehsil_ID={tehsil_id}"
                return True, msg, args
            except Exception as e:
                msg = f"[FAILED] Batch offset {offset} for Tehsil_ID={tehsil_id}: {e}"
                return False, msg, args

        # INITIAL PASS
        print(f"\nStarting batch download for {tehsil_name} ({len(initial_work_args)} batches)...")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(fetch_and_save_batch_with_status, args) for args in initial_work_args]
            for f in tqdm(as_completed(futures), total=len(futures), desc=f"Downloading {tehsil_name}", unit="batch"):
                success, msg, orig_args = f.result()
                print(msg)
                if not success:
                    failed_batches.append(orig_args)

        # RETRIES
        for retry in range(1, max_retries + 1):
            if not failed_batches:
                break
            print(f"\nRetrying {len(failed_batches)} failed batches for {tehsil_name} (Attempt {retry})...")
            still_failed = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                retry_args = []
                for args in failed_batches:
                    tehsil_id, offset, batch_size, base_url, tokens, proxies, tehsil_folder = args
                    retry_args.append((tehsil_id, offset, batch_size, base_url, tokens, proxies, tehsil_folder))
                futures = [executor.submit(fetch_and_save_batch_with_status, args) for args in retry_args]
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Retry {retry} {tehsil_name}",
                              unit="batch"):
                    success, msg, orig_args = f.result()
                    print(msg)
                    if not success:
                        still_failed.append(orig_args)
            failed_batches = still_failed

        if failed_batches:
            print(f"\nThe following batches FAILED after {max_retries} retries for {tehsil_name}:")
            for args in failed_batches:
                tehsil_id, offset, batch_size, *_ = args
                print(f"    Tehsil_ID={tehsil_id}, offset={offset}, batch_size={batch_size}")
        else:
            print(f"\nAll batches processed successfully for {tehsil_name} after {max_retries} retries.")

        print("Merging all batches into single GeoJSON for", tehsil_name)
        all_feats = load_all_batches(tehsil_folder)
        unique_features = list({json.dumps(f, sort_keys=True): f for f in all_feats}.values())
        with open(tehsil_raw_path, "w", encoding="utf-8") as f:
            json.dump({"type": "FeatureCollection", "features": unique_features}, f, indent=2)
        print(f"Raw features saved to {tehsil_raw_path}")

        # --- Delete the batches folder after use ---
        batches_dir = os.path.join(tehsil_folder, "batches")
        if os.path.exists(batches_dir):
            try:
                shutil.rmtree(batches_dir)
                print(f"Deleted batches folder: {batches_dir}")
            except Exception as e:
                print(f"Warning: Could not delete batches folder: {e}")

        # Use the selected district-root path via custom_output_folder
        group_and_save(unique_features, district_name, tehsil_name, custom_output_folder=tehsil_folder)
        zip_and_cleanup_centroids_and_bounds(district_name, tehsil_name, custom_output_folder=tehsil_folder)

    logging.info(f"Completed in {time.time()-t0:.1f}s")
