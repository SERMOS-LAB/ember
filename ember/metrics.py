"""
Evacuation metrics and summary reporting.
"""

import pandas as pd
import numpy as np


def compliance_rate(
    records: pd.DataFrame,
    zone_col: str = "ZoneType",
    category_col: str = "Category"
) -> pd.DataFrame:
    """
    Compute evacuation compliance rates by zone type.
    
    Compliant behavior generally includes SELE, FEUO, FEUW, PERE.
    Non-compliant is NER.
    
    Parameters
    ----------
    records : pd.DataFrame
        Classified evacuation metrics.
        
    Returns
    -------
    pd.DataFrame
        Rate of evacuation per zone type.
    """
    # Evacuees are categories 1,2,3,4,5. 
    # NER is 6. UR is 7.
    # Usually we count (Evacuated / Residing).
    
    evac_cats = ['SELE', 'PERE', 'FEUO', 'FEUW', 'SEFN']
    
    def calc_rate(group):
        total = len(group)
        # Exclude 'UR' from denominator if desired, but usually total = all valid inferred homes
        # For strictness, let's keep all.
        evacuated = group[category_col].isin(evac_cats).sum()
        return pd.Series({
            'Total_Residents': total,
            'Evacuated': evacuated,
            'Compliance_Rate': (evacuated / total) if total > 0 else 0.0
        })
        
    return records.groupby(zone_col).apply(calc_rate, include_groups=False).reset_index()


def dedi(
    records: pd.DataFrame,
    spatial_unit: str = "ZoneType"
) -> pd.DataFrame:
    """
    Damage-Evacuation Disparity Index (DEDI).
    
    A novel metric comparing physical fire proximity/damage to actual evacuation rates.
    For this prototype, it computes the micro-macro gap: difference between 
    expected evacuation rate (based on Order timing) and actual.
    
    Parameters
    ----------
    records : pd.DataFrame
        Needs spatial_unit column and 'Category'
    """
    # Simplified DEDI computation
    # In full form, this compares against an external dataset (like Facebook Movement)
    # Here we compute an internal DEDI: variance of compliance across regions inside the Order zone.
    
    order_only = records[records['ZoneType'] == 'Order']
    if order_only.empty:
        return pd.DataFrame()
        
    stats = compliance_rate(order_only, zone_col=spatial_unit, category_col="Category")
    
    # Calculate Disparity: how much each subregion deviates from the mean compliance
    mean_rate = stats['Compliance_Rate'].mean()
    stats['DEDI_Deviation'] = stats['Compliance_Rate'] - mean_rate
    
    # Standardize to 0-1 scale (absolute deviation relative to mean)
    stats['DEDI'] = np.abs(stats['DEDI_Deviation']) / mean_rate if mean_rate > 0 else 0
    
    return stats


def departure_curve(
    records: pd.DataFrame,
    start: str,
    end: str,
    freq: str = "1h",
    zone: str = "Order"
) -> pd.DataFrame:
    """
    Generate cumulative departure curves.
    
    Parameters
    ----------
    records : pd.DataFrame
        Requires DepartureDate and ZoneType columns.
        
    Returns
    -------
    pd.DataFrame
        Indexed by Datetime, column 'Cumulative_Pct'.
    """
    if 'DepartureDate' not in records.columns:
        raise ValueError("records must contain DepartureDate")
        
    df = records.copy()
    if zone:
        df = df[df['ZoneType'] == zone]
        
    # Filter to valid departures
    deps = pd.to_datetime(df['DepartureDate'].dropna())
    
    # Create time grid
    grid = pd.date_range(start=start, end=end, freq=freq)
    
    # Count cumulative sum
    # Sort deps
    deps = deps.sort_values()
    
    # Searchsorted finds the index where each grid time would be inserted to maintain order
    # It effectively gives the count of departures before or at each grid time
    counts = np.searchsorted(deps.values, grid.values)
    
    total = len(df) # total residents in this zone
    pct = (counts / total) * 100 if total > 0 else np.zeros_like(counts)
    
    out = pd.DataFrame({
        'Datetime': grid,
        'Cumulative_Count': counts,
        'Cumulative_Pct': pct
    }).set_index('Datetime')
    
    return out

def plot_departure_curve(
    curve_df: pd.DataFrame = None,
    df: pd.DataFrame = None,
    start_ref: pd.Timestamp = None,
    order_offsets: dict = None,
    night_shades: list = None,
    group_col: str = None,
    title: str = "Cumulative Departure Curve",
    output_path: str = None
):
    """
    Plot cumulative departure curves. 
    Can either plot a pre-computed curve_df (simple) or a raw metrics df (complex with day/night shading).
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Simple Mode
        if curve_df is not None:
            x = curve_df.index if isinstance(curve_df.index, pd.DatetimeIndex) else curve_df['Datetime']
            y = curve_df['Cumulative_Pct']
            ax.plot(x, y, linewidth=2, color='#F44336')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.xticks(rotation=45)
            ax.set_xlabel('Date / Time', fontsize=12)
        
        # Complex Mode (Fig 10 style)
        elif df is not None and start_ref is not None:
            evacuees = df[df['Category'] != 'NER'].copy()
            if 'DepartureDate' in evacuees.columns:
                try:
                    evacuees['DepartureDate'] = pd.to_datetime(evacuees['DepartureDate'])
                    # Safely remove timezone if present to avoid subtraction errors
                    if evacuees['DepartureDate'].dt.tz is not None:
                        evacuees['DepartureDate'] = evacuees['DepartureDate'].dt.tz_localize(None)
                    if getattr(start_ref, 'tzinfo', None) is not None:
                        start_ref = start_ref.tz_localize(None)
                        
                    evacuees['Hours_From_Start'] = (evacuees['DepartureDate'] - start_ref).dt.total_seconds() / 3600.0
                    subset = evacuees[(evacuees['Hours_From_Start'] >= 0) & (evacuees['Hours_From_Start'] <= 72)].copy()
                except Exception as e:
                    print(f"Error parsing DepartureDate timezone or calculating hours: {e}")
                    return
                
                if group_col and group_col in subset.columns:
                    groups = subset[group_col].unique()
                else:
                    subset['Group'] = 'All'
                    groups = ['All']
                    group_col = 'Group'
                    
                grand_total = len(subset)
                if grand_total > 0:
                    subset_sorted = subset.sort_values('Hours_From_Start')
                    subset_sorted['cumulative_pct'] = (np.arange(grand_total) + 1) / grand_total * 100
                    sns.lineplot(data=subset_sorted, x='Hours_From_Start', y='cumulative_pct', label=f"All Residents (n={grand_total})", color='black', linestyle='--', linewidth=2, alpha=0.6, ax=ax)
            
                colors_map = {
                    'Fully-compliant Evacuee Under Order': 'red', 'Ordered Evacuee': 'red', 'FEUO': 'red',
                    'Fully-compliant Evacuee Under Warning': 'orange', 'Evacuee under Warning': 'orange', 'FEUW': 'orange',
                    'Shadow Evacuee From Nearby': 'green', 'Shadow Evacuee': 'green', 'SEFN': 'green',
                    'Self-Evacuee Leaving Early': 'blue', 'Self-Evacuee': 'blue', 'SELE': 'blue',
                    'Partially-compliant Evacuee (Early Returner)': 'purple', 'PERE': 'purple',
                    'Order': '#F44336', 'Warning': '#FF9800', 'Buffer': '#2196F3',
                    'Palisades': '#E91E63', 'Eaton': '#9C27B0', 'East Altadena': 'dodgerblue', 'West Altadena': 'teal',
                    'All': 'black'
                }
                    
                for g in groups:
                    group_data = subset[subset[group_col] == g].sort_values('Hours_From_Start')
                    if group_data.empty: continue
                    group_total = len(group_data)
                    group_data['cumulative_count'] = np.arange(group_total) + 1
                    group_data['pct_of_group'] = group_data['cumulative_count'] / group_total * 100
                    sns.lineplot(data=group_data, x='Hours_From_Start', y='pct_of_group', label=f"{g} (n={group_total})", color=colors_map.get(g, None), ax=ax, linewidth=2, alpha=0.6)
                    
                if night_shades:
                    for (ns_start, ns_end) in night_shades:
                        # Convert Timestamps to hours-from-start_ref if needed
                        if isinstance(ns_start, pd.Timestamp):
                            ns_start = ns_start.tz_localize(None) if ns_start.tz is not None else ns_start
                            ns_start = (ns_start - start_ref).total_seconds() / 3600.0
                        if isinstance(ns_end, pd.Timestamp):
                            ns_end = ns_end.tz_localize(None) if ns_end.tz is not None else ns_end
                            ns_end = (ns_end - start_ref).total_seconds() / 3600.0
                        
                        if ns_start > 48:
                            continue
                            
                        draw_end = min(ns_end, 48)
                        ax.axvspan(ns_start, draw_end, alpha=0.10, color='grey', hatch='///')
                        mid = ns_start + (draw_end - ns_start) / 2
                        ax.text(mid, 2, "Night", ha='center', fontsize=8, color='black')
            
                # Extract automatic offsets if none provided
                if not order_offsets:
                    order_offsets = []
                    for col, label in [('WarningStart', 'First Warning'), ('OrderStart', 'First Evacuation Order')]:
                        if col in subset.columns:
                            valid_times = pd.to_datetime(subset[col]).dropna()
                            if not valid_times.empty:
                                first_time = valid_times.min()
                                offset = (first_time - start_ref).total_seconds() / 3600.0
                                order_offsets.append((label, offset))

                if order_offsets:
                    # Normalize dict → list of tuples
                    if isinstance(order_offsets, dict):
                        order_offsets = list(order_offsets.items())
                    # To avoid messy labels if there are duplicates, keep count
                    label_counts = {}
                    for label, offset in order_offsets:
                        count = label_counts.get(label, 0)
                        label_counts[label] = count + 1
                        display_text = label if count == 0 else f"{label} {count+1}"
                        ax.axvline(offset, color='purple', linestyle=':', alpha=0.8, label=display_text)
                        
                # Add Fire Start vertical line
                ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Fire Start')
                        
                ax.set_xlabel('Hours from Fire Start', fontsize=12)
                ax.set_xlim(0, 48)
                ax.set_ylim(0, 100)
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        ax.set_title(title, fontsize=14, weight='bold')
        ax.set_ylabel('% of Group Evacuated', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Warning: matplotlib/seaborn is required for plotting (`pip install matplotlib seaborn`).")


def plot_evacuation_composition(
    df: pd.DataFrame,
    group_col: str = "FireEvent",
    dual_panel: bool = False,
    title: str = "Evacuation Composition",
    output_path: str = None
):
    """
    Generate stacked bar chart of evacuation compositions with embedded data table.
    If dual_panel=True, plots two side-by-side charts: one with all categories, one excluding NER.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib and seaborn required.")
        return

    if df.empty or group_col not in df.columns:
        print(f"Group column {group_col} not found or dataframe empty.")
        return
        
    category_order = ['FEUO', 'FEUW', 'SELE', 'SEFN', 'PERE', 'NER', 'UR']
    colors = {
        'FEUO': '#d62728', 'FEUW': '#ff9896', 'SELE': '#ff7f0e',
        'SEFN': '#2ca02c', 'PERE': '#9467bd', 'NER': '#bcbd22', 'UR': '#7f7f7f'
    }
    
    def prep_data(subset, exclude_ner=False):
        if exclude_ner:
            subset = subset[subset['Category'] != 'NER']
        props = subset.groupby([group_col, 'Category'], observed=False).size().reset_index(name='count')
        totals = subset.groupby(group_col, observed=False)['ID'].nunique().reset_index(name='total')
        props = pd.merge(props, totals, on=group_col)
        props['pct'] = (props['count'] / props['total']) * 100
        
        pivot = props.pivot(index=group_col, columns='Category', values='pct').fillna(0)
        pivot = pivot.reindex(columns=[c for c in category_order if c in pivot.columns])
        
        pivot_counts = props.pivot(index=group_col, columns='Category', values='count').fillna(0).astype(int)
        pivot_counts = pivot_counts.reindex(columns=[c for c in category_order if c in pivot_counts.columns])
        return props, pivot, pivot_counts

    if dual_panel:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        panels = [
            (ax1, False, "Including NER (Non-Evacuating Resident)"),
            (ax2, True, "Excluding NER (Evacuees Only)")
        ]
        fig.suptitle(title, fontsize=16, weight='bold')
    else:
        fig, ax1 = plt.subplots(figsize=(8, 6))
        panels = [(ax1, False, title)]

    results = {}
    for ax, exclude_ner, p_title in panels:
        props, pivot, pivot_counts = prep_data(df, exclude_ner)
        
        if exclude_ner:
            sns.barplot(data=props, x=group_col, y='pct', hue='Category', palette=colors, ax=ax, edgecolor='black')
            ax.set_ylim(0, 115)
        else:
            pivot.plot(kind='bar', stacked=True, ax=ax, color=[colors.get(c, '#333333') for c in pivot.columns], width=0.6)
            ax.set_ylim(0, 100)
            
        ax.set_ylabel("Percentage (%)")
        ax.set_title(p_title)
        ax.tick_params(axis='x', rotation=0)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        df_formatted = pd.DataFrame(index=pivot.index, columns=pivot.columns)
        for region in pivot.index:
            for cat in pivot.columns:
                cnt = pivot_counts.loc[region, cat] if cat in pivot_counts.columns else 0
                pct = pivot.loc[region, cat] if cat in pivot.columns else 0
                df_formatted.loc[region, cat] = f"{int(cnt)} ({pct:.1f}%)"
                
        results['evacuees_only' if exclude_ner else 'all'] = df_formatted
        
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()
    
    return results


def _t2ll(x, y, level=14):
    import math
    map_size = 256 * (2 ** level)
    n = math.pi - 2 * math.pi * y / map_size
    lat_rad = math.atan(0.5 * (math.exp(n) - math.exp(-n)))
    lat = lat_rad * 180 / math.pi
    lon = 360 * x / map_size - 180
    return lat, lon


def _lat_lon_to_bing_tile(lat, lon, level=14):
    import math
    sin_lat = math.sin(lat * math.pi / 180)
    x = ((lon + 180) / 360) * 256 * (2 ** level)
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * 256 * (2 ** level)
    return int(x), int(y)


def plot_delay_map(
    df: pd.DataFrame,
    zones_gdf,
    title: str = "Spatial Distribution of Delay",
    output_path: str = None
):
    """
    Plot spatial heatmap of total evacuation hours delayed using Bing Tiles.
    zones_gdf should be a geopandas GeoDataFrame containing the fire boundaries.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import geopandas as gpd
        import contextily as ctx
        from shapely.geometry import box
    except ImportError:
        print("Warning: matplotlib, geopandas, contextily required.")
        return

    if zones_gdf.crs != "EPSG:3857": 
        zones_gdf = zones_gdf.to_crs("EPSG:3857")
        
    delayed_df = df[df['DelayHours'] > 0].copy() if 'DelayHours' in df.columns else pd.DataFrame()
    if delayed_df.empty or 'home_lat_4326' not in delayed_df.columns:
        print("Dataframe missing DelayHours or home coordinates.")
        return
    
    delayed_df['tile_pos'] = delayed_df.apply(lambda r: _lat_lon_to_bing_tile(r['home_lat_4326'], r['home_lon_4326']), axis=1)
    delayed_df['tile_x'] = delayed_df['tile_pos'].apply(lambda x: x[0])
    delayed_df['tile_y'] = delayed_df['tile_pos'].apply(lambda x: x[1])
    
    delayed_df['grid_x'] = delayed_df['tile_x'] // 256
    delayed_df['grid_y'] = delayed_df['tile_y'] // 256
    
    tile_stats = delayed_df.groupby(['grid_x', 'grid_y']).agg(total_delay=('DelayHours', 'sum'), user_count=('ID', 'count')).reset_index()
    
    def grid_to_poly(gx, gy, level=14):
        px, py = gx * 256, gy * 256
        lat1, lon1 = _t2ll(px, py, level)
        lat2, lon2 = _t2ll(px + 256, py + 256, level)
        return box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))

    geoms = [grid_to_poly(r.grid_x, r.grid_y) for r in tile_stats.itertuples()]
    tiles_gdf = gpd.GeoDataFrame(tile_stats, geometry=geoms, crs="EPSG:4326").to_crs(epsg=3857)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    try:
        ctx.add_basemap(ax, zoom=13, source=ctx.providers.CartoDB.Positron)
    except Exception:
        pass 
        
    tiles_gdf.plot(
        column='total_delay', ax=ax, cmap='inferno',
        norm=mcolors.LogNorm(vmin=tiles_gdf['total_delay'].min(), vmax=tiles_gdf['total_delay'].max()),
        alpha=0.9, edgecolor='none', legend=True,
        legend_kwds={'label': "Total Delay (Hours) - Log Scale", 'orientation': "horizontal", 'shrink': 0.7, 'pad': 0.05}
    )
    
    try:
        fire_boundary = zones_gdf.buffer(10).union_all().boundary
    except:
        fire_boundary = zones_gdf.unary_union.boundary
        
    gpd.GeoSeries([fire_boundary], crs=zones_gdf.crs).plot(ax=ax, color='red', linewidth=2.0, zorder=10, label='Evacuation Order Zone')
    
    minx, miny, maxx, maxy = zones_gdf.total_bounds
    buffer = 5000 
    ax.set_xlim(minx - buffer, maxx + buffer)
    ax.set_ylim(miny - buffer, maxy + buffer)
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    plt.figtext(0.5, 0.02, "Red Outline: Evacuation Order Zone. Tiles outside = Shadow Evacuation Delay.\\nMap shows Total Delay (Hours) per 2.4km Block (Log Scale).", ha="center", fontsize=11, style='italic', backgroundcolor='white')
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_delay_distribution(
    df: pd.DataFrame,
    group_col: str = "Category",
    title: str = "Evacuation Delay Distribution",
    output_path: str = None
):
    """
    Plot box + strip chart of DelayHours by group (Category or FireEvent).
    Only includes evacuees with DelayHours > 0.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib and seaborn required.")
        return

    delayed = df[(df['DelayHours'] > 0) & (df['DelayHours'].notna())].copy()
    if delayed.empty:
        print("No delay data to plot.")
        return

    cap = delayed['DelayHours'].quantile(0.95)
    delayed = delayed.dropna(subset=[group_col])
    order = sorted(delayed[group_col].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=delayed, x=group_col, y='DelayHours', order=order,
        palette='Set2', showfliers=False, ax=ax,
        boxprops=dict(alpha=0.6)
    )
    sns.stripplot(
        data=delayed, x=group_col, y='DelayHours', order=order,
        color='black', alpha=0.15, size=3, jitter=True, ax=ax
    )

    ax.set_ylim(0, cap * 1.1)
    ax.set_xlabel(group_col, fontsize=12)
    ax.set_ylabel('Delay (Hours)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_compliance_by_region(
    df: pd.DataFrame,
    region_col: str = "ZoneType",
    fire_col: str = "FireEvent",
    title: str = "Evacuation Compliance Rate by Region",
    output_path: str = None
):
    """
    Grouped bar chart of compliance rates across spatial sub-regions, colored by fire event.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib and seaborn required.")
        return

    if region_col not in df.columns:
        print(f"Column '{region_col}' not found.")
        return

    evac_cats = ['SELE', 'PERE', 'FEUO', 'FEUW', 'SEFN']
    group_cols = [region_col]
    if fire_col in df.columns:
        group_cols.append(fire_col)

    stats = df.groupby(group_cols, observed=False).apply(
        lambda g: pd.Series({
            'Total': len(g),
            'Evacuated': g['Category'].isin(evac_cats).sum(),
            'Rate': g['Category'].isin(evac_cats).mean() * 100
        }), include_groups=False
    ).reset_index()

    if stats.empty or 'Total' not in stats.columns:
        print(f"Not enough data to group by '{region_col}'. Check for NaN values.")
        return

    stats = stats[stats['Total'] >= 10]

    if stats.empty:
        print("Not enough data per region.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    hue = fire_col if fire_col in stats.columns else None
    sns.barplot(
        data=stats, x=region_col, y='Rate', hue=hue,
        palette='tab10', edgecolor='black', alpha=0.8, ax=ax
    )

    ax.set_xlabel(region_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Compliance Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    if hue:
        ax.legend(title=fire_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_temporal_heatmap(
    df: pd.DataFrame,
    fire_start: str = None,
    title: str = "Hourly Departure Heatmap",
    output_path: str = None
):
    """
    2D heatmap of departure counts: x = hour of day, y = date.
    Only includes rows with valid DepartureDate.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib and seaborn required.")
        return

    departures = df[df['DepartureDate'].notna()].copy()
    if departures.empty:
        print("No departure data.")
        return

    departures['DepartureDate'] = pd.to_datetime(departures['DepartureDate'])
    
    if departures['DepartureDate'].dt.tz is not None:
        departures['DepartureDate'] = departures['DepartureDate'].dt.tz_localize(None)

    departures['dep_date'] = departures['DepartureDate'].dt.date
    departures['dep_hour'] = departures['DepartureDate'].dt.hour

    if fire_start:
        fs = pd.Timestamp(fire_start)
        if getattr(fs, 'tzinfo', None) is not None:
            fs = fs.tz_localize(None)
        departures = departures[
            (departures['DepartureDate'] >= fs) &
            (departures['DepartureDate'] <= fs + pd.Timedelta(days=7))
        ]

    pivot = departures.groupby(['dep_date', 'dep_hour']).size().reset_index(name='count')
    heatmap_data = pivot.pivot(index='dep_date', columns='dep_hour', values='count').fillna(0)

    for h in range(24):
        if h not in heatmap_data.columns:
            heatmap_data[h] = 0
    heatmap_data = heatmap_data[sorted(heatmap_data.columns)]
    heatmap_data.index = [str(d) for d in heatmap_data.index]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        heatmap_data, cmap='YlOrRd', annot=True, fmt='.0f',
        linewidths=0.5, ax=ax, cbar_kws={'label': 'Number of Departures'}
    )
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Date', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()


def plot_return_timeline(
    df: pd.DataFrame,
    fire_start: str = None,
    group_col: str = "Category",
    title: str = "Return Behavior Timeline",
    output_path: str = None
):
    """
    Scatter/strip plot of return times (hours from fire start) by Category.
    Shows how quickly different evacuee groups return home.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: matplotlib and seaborn required.")
        return

    returners = df[df['ReturnDate'].notna()].copy()
    if returners.empty:
        print("No return data.")
        return

    returners['ReturnDate'] = pd.to_datetime(returners['ReturnDate'])
    if returners['ReturnDate'].dt.tz is not None:
        returners['ReturnDate'] = returners['ReturnDate'].dt.tz_localize(None)

    if fire_start:
        ref = pd.Timestamp(fire_start)
    elif 'OrderStart' in returners.columns:
        ref = pd.to_datetime(returners['OrderStart']).min()
    else:
        ref = returners['ReturnDate'].min()

    if getattr(ref, 'tzinfo', None) is not None:
        ref = ref.tz_localize(None)

    returners['return_hours'] = (returners['ReturnDate'] - ref).dt.total_seconds() / 3600.0
    returners = returners[returners['return_hours'] >= 0]
    returners = returners[returners['return_hours'] <= 720]

    category_order = ['SELE', 'PERE', 'FEUW', 'FEUO', 'SEFN']
    returners = returners[returners[group_col].isin(category_order)]

    colors = {
        'SELE': '#2196F3', 'PERE': '#4CAF50', 'FEUW': '#FF9800',
        'FEUO': '#F44336', 'SEFN': '#9C27B0'
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.stripplot(
        data=returners, x=group_col, y='return_hours', order=category_order,
        palette=colors, alpha=0.4, size=4, jitter=0.35, ax=ax
    )
    sns.boxplot(
        data=returners, x=group_col, y='return_hours', order=category_order,
        color='white', showfliers=False, ax=ax,
        boxprops=dict(alpha=0.7, edgecolor='black'),
        medianprops=dict(color='red', linewidth=2)
    )

    ax.axhline(y=24, color='gray', linestyle='--', alpha=0.5, label='24 hours')
    ax.axhline(y=72, color='gray', linestyle=':', alpha=0.5, label='3 days')
    ax.axhline(y=168, color='gray', linestyle='-.', alpha=0.5, label='1 week')

    ax.set_xlabel('Evacuee Category', fontsize=12)
    ax.set_ylabel('Hours from Fire Start to Return', fontsize=12)
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend(loc='upper right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()

