"""
Input/Output data loading module for EMBER.

Flexible loaders to ingest either pre-processed CSVs/GeoJSONs
or raw GPS data (MongoDB shards or Pandas dataframes).
"""

import os
import glob
import pickle
from pathlib import Path
from typing import Union, List, Tuple, Optional

import pandas as pd
import geopandas as gpd

from ._constants import EPSG_WGS84


class EmberInputError(Exception):
    """Raised when input data does not match the required schema."""
    pass


def load_evacuation_metrics(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load pre-computed evacuation metrics CSV.
    
    Parameters
    ----------
    path : str or Path
        Path to the evacuation_metrics.csv file.
        
    Returns
    -------
    pd.DataFrame
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise EmberInputError(f"Failed to load metrics from {path}: {e}")
        
    expected_cols = ['ID', 'Category', 'DelayHours', 'ZoneType']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        # We don't raise strict error here to allow loading raw un-classified DataFrames,
        # but we raise a warning if it looks like they wanted classified data, or we just
        # accept whatever is there. For flexibility, let's just return what we have.
        pass
        
    # Standardize dates if they exist
    date_cols = ['DepartureDate', 'ReturnDate', 'OrderStart', 'OrderEnd', 'FireStart']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            
    return df


def load_homes(path: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Load inferred home locations and normalize to EPSG:4326.
    
    Parameters
    ----------
    path : str or Path
        Path to the home.csv file.
        
    Returns
    -------
    gpd.GeoDataFrame
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise EmberInputError(f"Failed to load homes from {path}: {e}")
        
    if 'home_lon_4326' not in df.columns or 'home_lat_4326' not in df.columns:
        if 'home_lon' in df.columns and 'home_lat' in df.columns:
            # Assume 4326 if un-suffixed
            df['home_lon_4326'] = df['home_lon']
            df['home_lat_4326'] = df['home_lat']
        else:
            raise EmberInputError("Homes data requires longitude/latitude columns (e.g. 'home_lon_4326').")
            
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df['home_lon_4326'], df['home_lat_4326']),
        crs=EPSG_WGS84
    )
    return gdf


def load_fire_timeline(path: Union[str, Path, List[str]]) -> gpd.GeoDataFrame:
    """
    Load evacuation zones from one or multiple GeoJSON files.
    
    Parameters
    ----------
    path : str, Path, or list of str
        Path to a single .geojson file, a directory containing .geojson files,
        or a list of file paths.
        
    Returns
    -------
    gpd.GeoDataFrame
        Normalized to EPSG:4326.
    """
    files = []
    if isinstance(path, (str, Path)):
        p = Path(path)
        if p.is_dir():
            files = list(p.glob("*.geojson"))
        elif p.is_file():
            files = [p]
    else:
        files = [Path(f) for f in path]
        
    if not files:
        raise EmberInputError(f"No GeoJSON files found at path(s): {path}")
        
    gdfs = []
    for f in files:
        try:
            df = gpd.read_file(f)
            gdfs.append(df)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
            
    if not gdfs:
        raise EmberInputError("Could not parse any provided GeoJSON files.")
        
    combined = pd.concat(gdfs, ignore_index=True)
    
    # Ensure it's WGS84
    if combined.crs is None or combined.crs.to_epsg() != EPSG_WGS84:
        combined = combined.to_crs(EPSG_WGS84)
        
    # Deduplicate on zoneId and status if applicable
    subset = []
    if 'zoneId' in combined.columns:
        subset.append('zoneId')
    if 'most_extreme_status' in combined.columns:
        subset.append('most_extreme_status')
        
    if subset:
        # Keep the latest datetime for duplicate zones? Or drop purely identical ones.
        if 'datetime' in combined.columns:
            combined['datetime'] = pd.to_datetime(combined['datetime'])
            combined = combined.sort_values('datetime').drop_duplicates(subset=subset, keep='last')
        else:
            combined = combined.drop_duplicates(subset=subset)
            
    return combined


def load_shards(shard_dir: Union[str, Path], bbox: Optional[Tuple[float, float, float, float]] = None) -> pd.DataFrame:
    """
    Load raw GPS ping data from MongoDB pickled shard files (.pkl).
    
    Optionally filter to a specific bounding box (min_lon, min_lat, max_lon, max_lat).
    
    Parameters
    ----------
    shard_dir : str or Path
        Directory containing the shard_XX.pkl files.
    bbox : tuple of float, optional
        (min_lon, min_lat, max_lon, max_lat) in WGS84.
        
    Returns
    -------
    pd.DataFrame
        Combined raw pings from the study area.
    """
    shard_dir = Path(shard_dir)
    files = list(shard_dir.glob("shard_*.pkl"))
    
    if not files:
        raise EmberInputError(f"No shard_*.pkl files found in {shard_dir}")
        
    all_pings = []
    for f in files:
        try:
            with open(f, "rb") as pkl_file:
                # Based on prepare_la_fire_data.py schema: (local_residents, local_dau)
                _, dau_data = pickle.load(pkl_file)
                if dau_data:
                    df = pd.DataFrame(dau_data)
                    
                    if bbox:
                        minx, miny, maxx, maxy = bbox
                        mask = (
                            (df['LONG'] >= minx) & (df['LONG'] <= maxx) &
                            (df['LAT'] >= miny) & (df['LAT'] <= maxy)
                        )
                        df = df[mask]
                        
                    all_pings.append(df)
        except Exception as e:
            print(f"Warning: Failed to load shard {f}: {e}")
            
    if not all_pings:
        return pd.DataFrame()
        
    combined = pd.concat(all_pings, ignore_index=True)
    
    # Rename for ember standard schema (user_id, timestamp, lat, lon)
    col_mapping = {
        'GRID': 'user_id',
        'LAT': 'latitude',
        'LONG': 'longitude',
        'TIMESTAMP': 'timestamp_ms'
    }
    combined = combined.rename(columns=col_mapping)
    
    # Ensure there's a proper datetime column
    if 'timestamp_ms' in combined.columns and 'datetime' not in combined.columns:
        combined['datetime'] = pd.to_datetime(combined['timestamp_ms'], unit='ms', utc=True)
        
    return combined
