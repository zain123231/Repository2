import os
import pandas as pd

def rebuild_global_cities():
    cities_txt = "data/cities1000.txt"
    global_cities_csv = "data/global_cities.csv"
    
    if not os.path.exists(cities_txt):
        print(f"[ERROR] {cities_txt} not found.")
        return
        
    print("[LOG] Reading cities1000.txt...")
    # GeoNames format (tab separated)
    columns = [
        "geonameid", "name", "asciiname", "alternatenames", 
        "latitude", "longitude", "feature_class", "feature_code",
        "country_code", "cc2", "admin1_code", "admin2_code",
        "admin3_code", "admin4_code", "population", "elevation",
        "dem", "timezone", "modification_date"
    ]
    
    df = pd.read_csv(cities_txt, sep='\t', header=None, names=columns, low_memory=False, quoting=3)
    
    out_df = pd.DataFrame({
        "City": df["name"],
        "LAT": df["latitude"],
        "LON": df["longitude"],
        "CountryCode": df["country_code"],
        "feature_class": df["feature_class"],
        "population": df["population"]
    })
    
    # Fill NA for population and convert to int
    out_df["population"] = out_df["population"].fillna(0).astype(int)
    
    print(f"[LOG] Writing {len(out_df)} rows to {global_cities_csv}...")
    out_df.to_csv(global_cities_csv, index=False)
    print("[LOG] Done.")

if __name__ == "__main__":
    rebuild_global_cities()
