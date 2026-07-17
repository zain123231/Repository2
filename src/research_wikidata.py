import os
import requests
import time
import pandas as pd
from tqdm import tqdm
import math

OUTPUT_DIR = "data/iraq_wikidata_images"
OUTPUT_CSV = "data/iraq_wikidata_test.csv"
GLOBAL_TRAIN_CSV = "data/train.csv" # To check distance from training
MIN_DISTANCE_KM = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c

def fetch_wikidata_iraq_locations():
    """
    SPARQL query to get items in Iraq with an image and coordinates.
    """
    url = 'https://query.wikidata.org/sparql'
    query = """
    SELECT ?item ?itemLabel ?coord ?image WHERE {
      ?item wdt:P17 wd:Q796.      # Country: Iraq
      ?item wdt:P625 ?coord.      # Has coordinates
      ?item wdt:P18 ?image.       # Has image
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en,ar". }
    }
    LIMIT 2000
    """
    headers = {
        'User-Agent': 'GeoCLIP-ResearchBot/1.0 (local@localhost)',
        'Accept': 'application/sparql-results+json'
    }
    response = requests.get(url, params={'query': query}, headers=headers)
    if response.status_code == 200:
        data = response.json()
        results = []
        for binding in data['results']['bindings']:
            coord = binding['coord']['value'].replace('Point(', '').replace(')', '')
            lon, lat = map(float, coord.split())
            image_url = binding['image']['value']
            # Decode the URL to get the exact filename for Commons API
            filename = image_url.split('/')[-1]
            import urllib.parse
            filename = urllib.parse.unquote(filename)
            
            results.append({
                'id': binding['item']['value'].split('/')[-1],
                'name': binding['itemLabel']['value'],
                'lat': lat,
                'lon': lon,
                'image_url': image_url,
                'filename': filename
            })
        return results
    else:
        print(f"Error fetching Wikidata: {response.status_code}")
        return []

def fetch_commons_metadata(filenames):
    """
    Fetch license and author from Wikimedia Commons API using the filenames.
    """
    API_URL = "https://commons.wikimedia.org/w/api.php"
    metadata_dict = {}
    
    # Process in chunks of 50
    for i in range(0, len(filenames), 50):
        chunk = filenames[i:i+50]
        params = {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "extmetadata",
            "titles": "|".join([f"File:{f}" for f in chunk]),
            "format": "json"
        }
        
        try:
            response = requests.get(API_URL, params=params)
            data = response.json()
            if 'query' in data and 'pages' in data['query']:
                for page_id, page_data in data['query']['pages'].items():
                    if 'title' in page_data:
                        title = page_data['title'].replace("File:", "")
                        license_name = "Unknown"
                        author_name = "Unknown"
                        
                        if 'imageinfo' in page_data and len(page_data['imageinfo']) > 0:
                            info = page_data['imageinfo'][0]
                            if 'extmetadata' in info:
                                ext = info['extmetadata']
                                if 'LicenseShortName' in ext:
                                    license_name = ext['LicenseShortName']['value']
                                if 'Artist' in ext:
                                    import re
                                    author_raw = ext['Artist']['value']
                                    author_name = re.sub('<[^<]+>', '', author_raw)
                        
                        metadata_dict[title] = {"license": license_name, "author": author_name}
        except Exception as e:
            pass
            
    return metadata_dict

def download_image(url, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        return filepath
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return filepath
    except Exception:
        pass
    return None

def main():
    print("[LOG] Querying Wikidata for Iraqi test set candidates...")
    candidates = fetch_wikidata_iraq_locations()
    print(f"[LOG] Found {len(candidates)} raw candidates from Wikidata.")
    
    train_lats = []
    train_lons = []
    if os.path.exists(GLOBAL_TRAIN_CSV):
        train_df = pd.read_csv(GLOBAL_TRAIN_CSV)
        if 'LAT' in train_df.columns and 'LON' in train_df.columns:
            train_lats = train_df['LAT'].values
            train_lons = train_df['LON'].values
            print(f"[LOG] Loaded {len(train_lats)} training coordinates for distance filtering.")
            
    filtered_candidates = []
    for cand in tqdm(candidates, desc="Geographic Filtering"):
        # 1. Self-filtering: > 1km from already accepted test candidates
        too_close = False
        for acc in filtered_candidates:
            if haversine(cand['lat'], cand['lon'], acc['lat'], acc['lon']) < MIN_DISTANCE_KM:
                too_close = True
                break
                
        if too_close:
            continue
            
        # 2. Train-filtering: > 1km from ALL global training points
        # To avoid slow O(N) loop for each, we can vectorize if train_lats is huge, 
        # but for simplicity we do a quick numpy vector calculation
        if len(train_lats) > 0:
            lats_diff = np.radians(train_lats - cand['lat'])
            lons_diff = np.radians(train_lons - cand['lon'])
            lat1_rad = math.radians(cand['lat'])
            lat2_rad = np.radians(train_lats)
            
            a = np.sin(lats_diff/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(lons_diff/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances = 6371 * c
            if np.min(distances) < MIN_DISTANCE_KM:
                continue
                
        filtered_candidates.append(cand)
        
    print(f"[LOG] After geographic filtering (≥{MIN_DISTANCE_KM}km), {len(filtered_candidates)} candidates remain.")
    
    # Fetch Metadata
    print("[LOG] Fetching Authors & Licenses from Commons API...")
    filenames = [c['filename'] for c in filtered_candidates]
    metadata_dict = fetch_commons_metadata(filenames)
    
    valid_data = []
    for cand in tqdm(filtered_candidates, desc="Downloading Images"):
        # Create a unique, safe filename based on Wikidata ID instead of full url string
        safe_filename = f"{cand['id']}.jpg"
        filepath = download_image(cand['image_url'], safe_filename)
        
        if filepath:
            meta = metadata_dict.get(cand['filename'], {"license": "Unknown", "author": "Unknown"})
            valid_data.append({
                "IMG_PATH": safe_filename,
                "WIKIDATA_ID": cand['id'],
                "NAME": cand['name'],
                "LAT": cand['lat'],
                "LON": cand['lon'],
                "license": meta['license'],
                "author": meta['author']
            })
            
    df = pd.DataFrame(valid_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[LOG] Successfully built test set of {len(df)} images. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
