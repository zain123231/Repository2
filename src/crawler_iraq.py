import os
import requests
import json
import time
import pandas as pd
from tqdm import tqdm

API_URL = "https://commons.wikimedia.org/w/api.php"
OUTPUT_DIR = "data/iraq_training_images"
OUTPUT_CSV = "data/iraq_train.csv"

# Bounding box limits for Iraq roughly
MIN_LAT = 29.0
MAX_LAT = 37.5
MIN_LON = 38.8
MAX_LON = 48.6
STEP = 0.5 # About 50km steps, radius 10km (will miss some spots but fast to scrape)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_images_at_coord(lat, lon):
    params = {
        "action": "query",
        "list": "geosearch",
        "gscoord": f"{lat}|{lon}",
        "gsradius": 10000,  # 10km radius
        "gslimit": 500,
        "format": "json"
    }
    
    headers = {
        "User-Agent": "GeoCLIP-Iraq-FineTuning/1.0 (Contact: local@localhost)"
    }
    
    try:
        response = requests.get(API_URL, params=params, headers=headers)
        data = response.json()
        if 'query' in data and 'geosearch' in data['query']:
            return data['query']['geosearch']
    except Exception as e:
        print(f"Error fetching coord {lat},{lon}: {e}")
    return []

def get_image_urls(pageids):
    urls = []
    # Fetch in chunks of 50
    for i in range(0, len(pageids), 50):
        chunk = pageids[i:i+50]
        params = {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url",
            "pageids": "|".join(map(str, chunk)),
            "format": "json"
        }
        
        try:
            response = requests.get(API_URL, params=params)
            data = response.json()
            if 'query' in data and 'pages' in data['query']:
                pages = data['query']['pages']
                for page_id, page_data in pages.items():
                    if 'imageinfo' in page_data and len(page_data['imageinfo']) > 0:
                        url = page_data['imageinfo'][0]['url']
                        if url.lower().endswith(('.jpg', '.jpeg', '.png')):
                            urls.append((page_id, url))
        except Exception as e:
            print(f"Error fetching URLs: {e}")
    return urls

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
    except Exception as e:
        pass
    return None

def main():
    print("Starting massive grid search over Iraq via Wikimedia Commons...")
    
    all_locations = []
    
    # Create grid
    lats = [MIN_LAT + i*STEP for i in range(int((MAX_LAT-MIN_LAT)/STEP) + 1)]
    lons = [MIN_LON + i*STEP for i in range(int((MAX_LON-MIN_LON)/STEP) + 1)]
    
    total_grid = len(lats) * len(lons)
    print(f"Grid points to scan: {total_grid}")
    
    # We will just do a subset to test if the user wants full, we can adjust STEP
    pbar = tqdm(total=total_grid, desc="Scanning Grid")
    
    results = []
    
    for lat in lats:
        for lon in lons:
            places = fetch_images_at_coord(lat, lon)
            if places:
                # We found images
                page_ids = [p['pageid'] for p in places]
                urls = get_image_urls(page_ids)
                
                # Map URLs back to coordinates
                url_dict = dict(urls)
                
                for p in places:
                    pid = str(p['pageid'])
                    if pid in url_dict:
                        url = url_dict[pid]
                        filename = f"{pid}.jpg"
                        
                        # Add to list
                        results.append({
                            "id": pid,
                            "lat": p['lat'],
                            "lon": p['lon'],
                            "url": url,
                            "filename": filename
                        })
            
            pbar.update(1)
            time.sleep(0.1) # Be nice to API
            
    pbar.close()
    
    print(f"Found {len(results)} images in Iraq! Starting download...")
    
    valid_data = []
    for item in tqdm(results, desc="Downloading Images"):
        filepath = download_image(item['url'], item['filename'])
        if filepath:
            valid_data.append({
                "filename": item['filename'],
                "lat": item['lat'],
                "lon": item['lon']
            })
            
    df = pd.DataFrame(valid_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Downloaded {len(df)} images successfully and saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
