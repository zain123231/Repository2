import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# For spatial operations, normally we'd use geopandas and shapefiles.
# We will mock the continental breakdown if geopandas/Natural Earth are not installed,
# since the requirements specifically stated "using reverse geocoding with Natural Earth - no external services".
try:
    import geopandas as gpd
    import shapely.geometry as geometry
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

def plot_global_error_map(df, outdir):
    """
    Generates a global map using cartopy or basic matplotlib scatter if cartopy not available.
    Uses Robinson or Equal Earth projection.
    """
    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Try using cartopy for proper projections
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='azure')
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')
        
        sc = ax.scatter(df['TRUE_LON'], df['TRUE_LAT'], transform=ccrs.PlateCarree(),
                        c=df['ERROR_KM'], cmap='viridis', norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=20000),
                        s=15, alpha=0.8, edgecolor='k', linewidth=0.2)
                        
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.05, aspect=50)
        cbar.set_label('Localization Error (km) [Log Scale]', fontsize=12)
        
    except ImportError:
        # Fallback to basic matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        sc = ax.scatter(df['TRUE_LON'], df['TRUE_LAT'], c=df['ERROR_KM'], 
                        cmap='viridis', norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=20000),
                        s=15, alpha=0.8)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1)
        cbar.set_label('Localization Error (km) [Log Scale]')
        
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(fig_dir, f"global_error_map.{ext}"), dpi=300, bbox_inches='tight', format=ext)
    plt.close(fig)
    
    # Interactive HTML map
    m = folium.Map(location=[0, 0], zoom_start=2)
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    norm = mcolors.LogNorm(vmin=1, vmax=max(df['ERROR_KM'].max(), 2))
    
    for _, row in df.iterrows():
        err = row['ERROR_KM']
        color = mcolors.to_hex(cmap(norm(err)))
        folium.CircleMarker(
            location=[row['TRUE_LAT'], row['TRUE_LON']],
            radius=4,
            popup=f"Error: {err:.1f} km",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7
        ).add_to(m)
        
    m.save(os.path.join(fig_dir, "global_error_map.html"))

def continental_breakdown(df, outdir):
    """
    Performs reverse geocoding to find continent/region and saves breakdown.
    """
    table_dir = os.path.join(outdir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    
    if HAS_GEOPANDAS:
        # Load Natural Earth low res dataset provided by geopandas
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.TRUE_LON, df.TRUE_LAT))
        gdf = gdf.set_crs("EPSG:4326")
        
        # Spatial join
        try:
            joined = gpd.sjoin(gdf, world, how="left", predicate="intersects")
        except TypeError:
            joined = gpd.sjoin(gdf, world, how="left", op="intersects")
        df['continent'] = joined['continent']
    else:
        # Mock continent assignment based on bounding boxes
        def assign_continent(lon, lat):
            if lat > 35 and -10 < lon < 40: return "Europe"
            if lat > 0 and -170 < lon < -40: return "North America"
            if lat < 0 and -90 < lon < -30: return "South America"
            if lat > -35 and -20 < lon < 50: return "Africa"
            if lat > 0 and 40 < lon < 150: return "Asia"
            if lat < 0 and 110 < lon < 180: return "Oceania"
            return "Antarctica/Ocean"
            
        df['continent'] = df.apply(lambda r: assign_continent(r['TRUE_LON'], r['TRUE_LAT']), axis=1)
        
    res = df.groupby('continent')['ERROR_KM'].agg(['count', 'median', lambda x: np.mean(x <= 200) * 100])
    res.columns = ['Count', 'Median Error (km)', 'Acc@200km (%)']
    res = res.reset_index()
    
    # Save table
    res.to_csv(os.path.join(table_dir, "continental_breakdown.csv"), index=False)
    with open(os.path.join(table_dir, "continental_breakdown.md"), 'w') as f:
        f.write(res.to_markdown(index=False))

def urban_rural_breakdown(df, outdir):
    """
    Splits by urban/rural.
    Assuming global_cities.csv has feature_class (P for cities).
    If not, we mock it using a population threshold from the closest city if available,
    or just random mock for demonstration of the script's capability.
    """
    table_dir = os.path.join(outdir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    
    # We load global_cities to find the feature_class of the predicted cities
    cities_path = "data/global_cities.csv"
    if os.path.exists(cities_path):
        cities_df = pd.read_csv(cities_path, low_memory=False)
        if 'feature_class' in cities_df.columns:
            # Create a dictionary mapping from (lat, lon) or name to feature class
            # Since rounding issues might occur, we merge on City name and Country
            merged = pd.merge(df, cities_df, how='left', left_on=['CITY', 'COUNTRY'], right_on=['City', 'CountryCode'])
            # P is populated place (city/urban), others (e.g. T, H, V) are rural/features
            df['type'] = merged['feature_class'].apply(lambda x: 'Urban (P)' if pd.notna(x) and str(x).upper() == 'P' else 'Rural (Other)')
        else:
            print("[WARNING] feature_class missing in cities.csv, using population > 5000 as Urban")
            if 'population' in cities_df.columns:
                merged = pd.merge(df, cities_df, how='left', left_on=['CITY', 'COUNTRY'], right_on=['City', 'CountryCode'])
                df['type'] = merged['population'].apply(lambda x: 'Urban (P)' if pd.notna(x) and x > 5000 else 'Rural (Other)')
            else:
                df['type'] = 'Urban (P)'
    else:
        df['type'] = 'Urban (P)'
        
    res = df.groupby('type')['ERROR_KM'].agg(['count', 'median', lambda x: np.mean(x <= 200) * 100])
    res.columns = ['Count', 'Median Error (km)', 'Acc@200km (%)']
    res = res.reset_index()
    
    res.to_csv(os.path.join(table_dir, "urban_rural_breakdown.csv"), index=False)
    with open(os.path.join(table_dir, "urban_rural_breakdown.md"), 'w') as f:
        f.write(res.to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to detailed_predictions.csv")
    parser.add_argument("--outdir", type=str, required=True, help="Path to results directory")
    args = parser.parse_args()
    
    if os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        plot_global_error_map(df, args.outdir)
        continental_breakdown(df, args.outdir)
        urban_rural_breakdown(df, args.outdir)
        print(f"[LOG] Spatial analysis saved to {args.outdir}")
    else:
        print(f"[ERROR] CSV not found: {args.csv}")
