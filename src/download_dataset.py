import os
import csv
import urllib.request

def download_sample_dataset():
    """
    Downloads a small set of real geo-tagged images (landmarks) to test the pipeline
    without needing gigabytes of data.
    """
    data_dir = "data"
    images_dir = os.path.join(data_dir, "images")
    csv_file = os.path.join(data_dir, "train.csv")
    
    os.makedirs(images_dir, exist_ok=True)
    
    # Dataset of real images (Wikimedia Commons direct URLs) + LAT/LON
    sample_data = [
        {
            "filename": "eiffel_tower.jpg",
            "url": "https://upload.wikimedia.org/wikipedia/commons/8/85/Tour_Eiffel_Wikimedia_Commons_%28cropped%29.jpg",
            "lat": 48.8584,
            "lon": 2.2945
        },
        {
            "filename": "statue_of_liberty.jpg",
            "url": "https://upload.wikimedia.org/wikipedia/commons/a/a1/Statue_of_Liberty_7.jpg",
            "lat": 40.6892,
            "lon": -74.0445
        },
        {
            "filename": "colosseum.jpg",
            "url": "https://upload.wikimedia.org/wikipedia/commons/d/d8/Colosseum_in_Rome-April_2007-1-_copie_2B.jpg",
            "lat": 41.8902,
            "lon": 12.4922
        },
        {
            "filename": "sydney_opera_house.jpg",
            "url": "https://upload.wikimedia.org/wikipedia/commons/a/a0/Sydney_Australia._%2821339175489%29.jpg",
            "lat": -33.8568,
            "lon": 151.2153
        },
        {
            "filename": "taj_mahal.jpg",
            "url": "https://upload.wikimedia.org/wikipedia/commons/b/bd/Taj_Mahal%2C_Agra%2C_India_edit3.jpg",
            "lat": 27.1751,
            "lon": 78.0421
        },
        {
            "filename": "machu_picchu.jpg",
            "url": "https://upload.wikimedia.org/wikipedia/commons/e/eb/Machu_Picchu%2C_Peru.jpg",
            "lat": -13.1631,
            "lon": -72.5450
        },
        {
            "filename": "great_wall_china.jpg",
            "url": "https://upload.wikimedia.org/wikipedia/commons/2/23/The_Great_Wall_of_China_at_Jinshanling-edit.jpg",
            "lat": 40.4319,
            "lon": 116.5704
        }
    ]
    
    print("[LOG] Downloading Sample Real Dataset...")
    
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["IMG_PATH", "LAT", "LON"])
        
        for idx, item in enumerate(sample_data):
            filepath = os.path.join(images_dir, item["filename"])
            print(f"[{idx+1}/{len(sample_data)}] Downloading {item['filename']}...")
            try:
                req = urllib.request.Request(item["url"], headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(filepath, 'wb') as out_file:
                    out_file.write(response.read())
                writer.writerow([item["filename"], item["lat"], item["lon"]])
            except Exception as e:
                print(f"[ERROR] Failed to download {item['filename']}: {e}")
                
    print(f"\n[LOG] Dataset successfully downloaded!")
    print(f"[LOG] Images saved in: {images_dir}")
    print(f"[LOG] Annotations saved in: {csv_file}")
    
if __name__ == "__main__":
    download_sample_dataset()
