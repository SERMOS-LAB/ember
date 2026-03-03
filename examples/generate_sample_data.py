import os
import pandas as pd
import geopandas as gpd
import shutil

# Paths
BASE_DIR = "/Users/mustafasameen/UFL Dropbox/Mustafa Sameen/projects/LA Fire"
DATA_IN = os.path.join(BASE_DIR, "fire_analysis/output")
GEO_IN = os.path.join(BASE_DIR, "Fire_Timeline")

OUT_DIR = os.path.join(BASE_DIR, "ember/data_sample")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"Generating EMBER sample data in {OUT_DIR}...")

# 1. Sample Home locations (EPSG:4326 normalized)
homes_path = os.path.join(DATA_IN, "home.csv")
if os.path.exists(homes_path):
    print("Loading homes...")
    df_homes = pd.read_csv(homes_path)
    # Take a 1% random sample to keep it lightweight (e.g. 1000 users)
    df_sample = df_homes.sample(min(1000, len(df_homes)), random_state=42)
    # Keep only the necessary columns for EMBER to demonstrate clean input
    df_sample = df_sample[['ID', 'home_lon_4326', 'home_lat_4326', 'num_nights', 'stay_time']]
    out_homes = os.path.join(OUT_DIR, "sample_homes.csv")
    df_sample.to_csv(out_homes, index=False)
    print(f" -> Saved {out_homes} ({len(df_sample)} rows)")

# 2. Sample Evacuation Metrics
metrics_path = os.path.join(DATA_IN, "evacuation_metrics.csv")
if os.path.exists(metrics_path):
    print("Loading metrics...")
    df_metrics = pd.read_csv(metrics_path)
    # Match the sample homes
    sampled_ids = df_sample['ID'].unique() if os.path.exists(homes_path) else df_metrics['ID'].sample(1000)
    df_metrics_sample = df_metrics[df_metrics['ID'].isin(sampled_ids)]
    
    # Optional: Keep only the input columns that EMBER would USE for taxonomy (rather than the outputs it will generate)
    # In practice, users provide GPS stops. We'll provide the raw metrics they'd want to classify
    keep_cols = ['ID', 'DepartureDate', 'ReturnDate', 'OrderStart', 'ZoneType']
    df_metrics_small = df_metrics_sample[[c for c in keep_cols if c in df_metrics_sample.columns]]
    out_metrics = os.path.join(OUT_DIR, "sample_evacuation_metrics.csv")
    df_metrics_small.to_csv(out_metrics, index=False)
    print(f" -> Saved {out_metrics} ({len(df_metrics_small)} rows)")

# 3. Fire Timeline (GeoJSON)
if os.path.exists(GEO_IN):
    print("Copying 2 sample GeoJSON timelines...")
    shutil.copytree(GEO_IN, os.path.join(OUT_DIR, "Fire_Timeline_Sample"), dirs_exist_ok=True)
    # Just keep a couple files
    for root, dirs, files in os.walk(os.path.join(OUT_DIR, "Fire_Timeline_Sample")):
        # just keep the first two
        kept = 0
        for f in files:
            if f.endswith('.geojson'):
                if kept < 2:
                    kept += 1
                else:
                    os.remove(os.path.join(root, f))
    print(f" -> Copied sample Fire_Timeline to {os.path.join(OUT_DIR, 'Fire_Timeline_Sample')}")

print("\nDone! Users can now test EMBER using `ember/data_sample/`.")
