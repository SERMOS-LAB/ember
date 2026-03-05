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


# ==========================================================================
# DESTINATION ANALYSIS PLOTS
# ==========================================================================

def plot_distance_histogram(
    destinations: pd.DataFrame,
    *,
    distance_col: str = "eu_distance_km",
    group_col: str | None = None,
    max_km: float = 300,
    bin_width: float = 10,
    title: str = "Travel Distance Histogram with Cumulative Distribution",
    output_path: str = None,
):
    """
    Histogram of travel distances with overlaid CDF.

    When ``group_col`` is None → single histogram + CDF.
    When ``group_col`` is provided → overlaid histogram + CDF per group.
    """
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(12, 7))
    bins = np.arange(0, max_km + bin_width, bin_width)

    if group_col is None or group_col not in destinations.columns:
        data = destinations[distance_col].dropna()
        ax1.hist(data, bins=bins, alpha=0.5, color='#7986cb', edgecolor='white', label='Number of destinations')
        ax2 = ax1.twinx()
        sorted_d = np.sort(data)
        cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
        ax2.plot(sorted_d, cdf, color='#1a237e', linewidth=2, marker='', label='Cumulative distribution')
        ax2.set_ylabel('Cumulative Distribution', fontsize=13)
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)
    else:
        ax2 = ax1.twinx()
        colors = plt.cm.Set2.colors
        groups = destinations[group_col].dropna().unique()
        for idx, g in enumerate(sorted(groups)):
            data = destinations.loc[destinations[group_col] == g, distance_col].dropna()
            c = colors[idx % len(colors)]
            ax1.hist(data, bins=bins, alpha=0.4, color=c, edgecolor='white', label=f'{g}: count')
            sorted_d = np.sort(data)
            cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            ax2.plot(sorted_d, cdf, color=c, linewidth=2, marker='o', markersize=2, label=f'{g}: CDF')
        ax2.set_ylabel('Cumulative Distribution', fontsize=13)
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

    ax1.set_xlabel('Euclidean Distance (km)', fontsize=13)
    ax1.set_ylabel('Number of Destinations', fontsize=13)
    ax1.set_xlim(0, max_km)
    ax1.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_distance_decay(
    destinations: pd.DataFrame,
    *,
    distance_col: str = "eu_distance_km",
    group_col: str = "Category",
    max_km: float = 300,
    title: str = "Travel Distance-Decay by Group",
    output_path: str = None,
):
    """
    Distance-decay (reverse CDF) curves by evacuee group.

    Shows the ratio of destinations beyond a given distance threshold.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'Warned & ordered evacuee': 'red', 'Self evacuee': 'green', 'Shadow evacuee': 'blue'}
    default_colors = plt.cm.Set1.colors

    groups = destinations[group_col].dropna().unique()
    for idx, g in enumerate(sorted(groups)):
        data = destinations.loc[destinations[group_col] == g, distance_col].dropna().values
        if len(data) == 0:
            continue
        thresholds = np.arange(0, max_km + 1, 5)
        ratios = [np.sum(data >= t) / len(data) for t in thresholds]
        c = colors.get(g, default_colors[idx % len(default_colors)])
        ax.plot(thresholds, ratios, linewidth=2, marker='o', markersize=3, color=c, label=str(g))

    ax.set_xlabel('Euclidean Distance (km)', fontsize=13)
    ax.set_ylabel('Ratio of Destinations', fontsize=13)
    ax.set_xlim(0, max_km)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_destination_types(
    destinations: pd.DataFrame,
    *,
    type_col: str = "dest_type",
    distance_col: str = "eu_distance_km",
    distance_threshold: float | None = None,
    threshold_mode: str = "all",
    title: str = "Destination Types",
    output_path: str = None,
):
    """
    Stacked or pie chart of destination types.

    Parameters
    ----------
    distance_threshold : float or None
        If set, filter by distance.
    threshold_mode : str
        ``'leq'`` for ≤ threshold, ``'gt'`` for > threshold, ``'all'`` for no filter.
    """
    import matplotlib.pyplot as plt

    df = destinations.copy()
    if distance_threshold is not None:
        if threshold_mode == "leq":
            df = df[df[distance_col] <= distance_threshold]
        elif threshold_mode == "gt":
            df = df[df[distance_col] > distance_threshold]

    if df.empty or type_col not in df.columns:
        print(f"No data to plot for {title}")
        return

    counts = df[type_col].value_counts()
    pcts = (counts / counts.sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors_map = {
        'residential': '#4fc3f7', 'hotel/motel': '#ff8a65',
        'commercial': '#81c784', 'public': '#ba68c8',
        'road': '#ffb74d', 'other': '#90a4ae',
    }
    pie_colors = [colors_map.get(t, '#bdbdbd') for t in pcts.index]
    wedges, texts, autotexts = ax.pie(
        pcts.values, labels=pcts.index, autopct='%1.1f%%',
        colors=pie_colors, textprops={'fontsize': 11}
    )
    for a in autotexts:
        a.set_fontsize(10)
    ax.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

    return pd.DataFrame({'type': counts.index, 'count': counts.values, 'pct': pcts.values})


def plot_dest_type_comparison(
    destinations: pd.DataFrame,
    *,
    type_col: str = "dest_type",
    distance_col: str = "eu_distance_km",
    threshold_km: float = 30.0,
    title: str = "Destination Types: ≤30 km vs >30 km",
    output_path: str = None,
):
    """
    Side-by-side bar comparison of destination types ≤ threshold vs > threshold.
    """
    import matplotlib.pyplot as plt

    near = destinations[destinations[distance_col] <= threshold_km]
    far = destinations[destinations[distance_col] > threshold_km]

    if near.empty and far.empty:
        print("No data to plot.")
        return

    all_types = sorted(destinations[type_col].dropna().unique())
    near_pcts = near[type_col].value_counts(normalize=True).reindex(all_types, fill_value=0) * 100
    far_pcts = far[type_col].value_counts(normalize=True).reindex(all_types, fill_value=0) * 100

    x = np.arange(len(all_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, near_pcts.values, width, label=f'≤ {threshold_km} km', color='#4fc3f7')
    ax.bar(x + width / 2, far_pcts.values, width, label=f'> {threshold_km} km', color='#ff8a65')
    ax.set_xticks(x)
    ax.set_xticklabels(all_types, rotation=30, ha='right', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=13)
    ax.legend(fontsize=12)
    ax.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_group_dest_heatmap(
    destinations: pd.DataFrame,
    *,
    group_col: str = "Category",
    type_col: str = "dest_type",
    distance_col: str = "eu_distance_km",
    metric: str = "pct",
    title: str = "Trips by Group and Destination Type",
    output_path: str = None,
):
    """
    Annotated heatmap: group × destination type.

    ``metric`` controls the cell values:

        ``'pct'`` → percent of trips.
        ``'median_distance'`` → median distance in km.
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    df = destinations.dropna(subset=[group_col, type_col])
    if df.empty:
        print("No data to plot.")
        return

    groups = sorted(df[group_col].unique())
    types = sorted(df[type_col].unique())

    matrix = np.zeros((len(groups), len(types)))
    for i, g in enumerate(groups):
        sub = df[df[group_col] == g]
        for j, t in enumerate(types):
            sub_t = sub[sub[type_col] == t]
            if metric == "pct":
                matrix[i, j] = round(len(sub_t) / max(len(sub), 1) * 100, 1)
            else:
                matrix[i, j] = round(sub_t[distance_col].median(), 1) if len(sub_t) > 0 else 0

    fig, ax = plt.subplots(figsize=(10, max(5, len(groups) * 0.8 + 1)))
    cmap = plt.cm.Blues if metric == "pct" else plt.cm.GnBu
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=30, ha='right', fontsize=11)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=11)

    for i in range(len(groups)):
        for j in range(len(types)):
            val = matrix[i, j]
            text_color = 'white' if val > matrix.max() * 0.6 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=11, color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    label = 'Percent of Trips (%)' if metric == "pct" else 'Median Distance (km)'
    ax.set_title(f'{title}\n({label})', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_od_map(
    destinations: pd.DataFrame,
    fire_zones=None,
    *,
    order_zones=None,
    warning_zones=None,
    home_lat_col: str = "home_lat_4326",
    home_lon_col: str = "home_lon_4326",
    dest_lat_col: str = "dest_lat",
    dest_lon_col: str = "dest_lon",
    title: str = "Origin-Destination Movement",
    max_trips: int | None = 500,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    zoom_percentile: float = 90,
    figsize: tuple = (14, 12),
    output_path: str = None,
):
    """
    Map showing O-D pairs as lines from home to destination.

    Uses contextily tile basemap for geographic context.
    Automatically samples trips if the dataset is too large.

    Parameters
    ----------
    fire_zones : GeoDataFrame or None
        Combined fire/evacuation zones (single color). Ignored if
        ``order_zones`` or ``warning_zones`` are provided.
    order_zones : GeoDataFrame or None
        Evacuation Order zones (rendered in dark red).
    warning_zones : GeoDataFrame or None
        Evacuation Warning zones (rendered in pink/magenta).
    max_trips : int or None
        Maximum number of O-D lines to draw. ``None`` draws all.
    xlim : tuple or None
        ``(lon_min, lon_max)`` to override automatic extent.
    ylim : tuple or None
        ``(lat_min, lat_max)`` to override automatic extent.
    zoom_percentile : float
        Percentile (0–100) for auto-clipping outlier destinations
        (default 90). Set to 100 for no clipping.
    figsize : tuple
        Figure size ``(width, height)``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError:
        print("geopandas and shapely are required for plot_od_map")
        return

    df = destinations.dropna(subset=[home_lat_col, home_lon_col, dest_lat_col, dest_lon_col]).copy()
    if df.empty:
        print("No data to plot.")
        return

    # ---- Density control: sample if too many trips ----
    n_total = len(df)
    sampled = False
    if max_trips is not None and n_total > max_trips:
        df = df.sample(n=max_trips, random_state=42)
        sampled = True

    # Build GeoDataFrames in EPSG:4326
    lines = [
        LineString([
            (row[home_lon_col], row[home_lat_col]),
            (row[dest_lon_col], row[dest_lat_col])
        ])
        for _, row in df.iterrows()
    ]
    gdf_lines = gpd.GeoDataFrame(df, geometry=lines, crs="EPSG:4326")
    origins = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[home_lon_col], df[home_lat_col]), crs="EPSG:4326"
    )
    dests = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df[dest_lon_col], df[dest_lat_col]), crs="EPSG:4326"
    )

    # ---- Project to Web Mercator for basemap ----
    gdf_lines_wm = gdf_lines.to_crs(epsg=3857)
    origins_wm = origins.to_crs(epsg=3857)
    dests_wm = dests.to_crs(epsg=3857)

    # ---- Compute bounds FIRST ----
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    if xlim is not None and ylim is not None:
        x0, y0 = transformer.transform(xlim[0], ylim[0])
        x1, y1 = transformer.transform(xlim[1], ylim[1])
    elif xlim is not None:
        x0, _ = transformer.transform(xlim[0], 0)
        x1, _ = transformer.transform(xlim[1], 0)
        y0, y1 = dests_wm.total_bounds[1], dests_wm.total_bounds[3]
    elif ylim is not None:
        _, y0 = transformer.transform(0, ylim[0])
        _, y1 = transformer.transform(0, ylim[1])
        x0, x1 = dests_wm.total_bounds[0], dests_wm.total_bounds[2]
    else:
        # Auto-clip to percentile bounds
        all_lons = pd.concat([df[home_lon_col], df[dest_lon_col]])
        all_lats = pd.concat([df[home_lat_col], df[dest_lat_col]])
        lo_p = (100 - zoom_percentile) / 2
        hi_p = 100 - lo_p
        lon_lo, lon_hi = np.percentile(all_lons, [lo_p, hi_p])
        lat_lo, lat_hi = np.percentile(all_lats, [lo_p, hi_p])
        pad_lon = max(0.2, (lon_hi - lon_lo) * 0.12)
        pad_lat = max(0.2, (lat_hi - lat_lo) * 0.12)
        x0, y0 = transformer.transform(lon_lo - pad_lon, lat_lo - pad_lat)
        x1, y1 = transformer.transform(lon_hi + pad_lon, lat_hi + pad_lat)

    # ---- Create figure with correct proportions ----
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    fig_w = figsize[0]
    fig_h = max(6, min(20, fig_w * (dy / dx) + 1.0)) if dx > 0 else figsize[1]
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    # ---- Evacuation zones (Order / Warning / combined) ----
    has_separate_zones = order_zones is not None or warning_zones is not None
    if has_separate_zones:
        if warning_zones is not None:
            try:
                wz = warning_zones.to_crs(epsg=3857)
                wz.plot(ax=ax, color='#f48fb1', alpha=0.30, edgecolor='#c2185b',
                        linewidth=0.6, zorder=2)
            except Exception:
                pass
        if order_zones is not None:
            try:
                oz = order_zones.to_crs(epsg=3857)
                oz.plot(ax=ax, color='#ef5350', alpha=0.40, edgecolor='#b71c1c',
                        linewidth=0.8, zorder=2)
            except Exception:
                pass
    elif fire_zones is not None:
        try:
            fz = fire_zones.to_crs(epsg=3857)
            fz.plot(ax=ax, color='#ef5350', alpha=0.35, edgecolor='darkred',
                    linewidth=0.8, zorder=2)
        except Exception:
            pass

    # ---- O-D lines and points ----
    n = len(df)
    line_alpha = max(0.08, min(0.5, 200 / n))
    point_alpha = max(0.2, min(0.7, 300 / n))
    point_size = max(1, min(8, 500 / n))

    gdf_lines_wm.plot(ax=ax, color='#ffb74d', alpha=line_alpha, linewidth=0.3, zorder=3)
    origins_wm.plot(ax=ax, color='#2e7d32', markersize=point_size, alpha=point_alpha, zorder=4)
    dests_wm.plot(ax=ax, color='#1565c0', markersize=point_size, alpha=point_alpha, zorder=5)

    # ---- Add basemap tiles (no labels for clean look) ----
    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom='auto')
    except (ImportError, AttributeError):
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom='auto')
        except Exception:
            ax.set_facecolor('#f5f5f5')
    except Exception:
        ax.set_facecolor('#f5f5f5')

    # ---- Manual legend ----
    legend_handles = [
        Line2D([0], [0], color='#ffb74d', linewidth=1.5, label='O-D Trip'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2e7d32',
               markersize=6, label='Origin (Home)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1565c0',
               markersize=6, label='Destination'),
    ]
    if has_separate_zones:
        if order_zones is not None:
            legend_handles.append(
                Patch(facecolor='#ef5350', alpha=0.40, edgecolor='#b71c1c',
                      label='Evacuation Order'))
        if warning_zones is not None:
            legend_handles.append(
                Patch(facecolor='#f48fb1', alpha=0.30, edgecolor='#c2185b',
                      label='Evacuation Warning'))
    elif fire_zones is not None:
        legend_handles.append(
            Patch(facecolor='#ef5350', alpha=0.35, edgecolor='darkred',
                  label='Fire/Evac Zone'))

    subtitle = f'({n_total:,} trips' + (f', showing {max_trips:,} sampled)' if sampled else ')')
    ax.legend(handles=legend_handles, fontsize=10, loc='lower left',
              framealpha=0.9, edgecolor='gray')
    ax.set_title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()



def plot_evacuee_group_summary(
    destinations: pd.DataFrame,
    *,
    group_col: str = "Category",
    id_col: str = "ID",
    title: str = "Evacuee Groups and Destinations",
    output_path: str = None,
):
    """
    Dual-axis chart: percentage of evacuee type (bar) + mean/median
    number of destinations per person (line).
    """
    import matplotlib.pyplot as plt

    # Count unique evacuees and destinations per group
    user_groups = destinations.drop_duplicates(subset=[id_col])
    grp_counts = user_groups[group_col].value_counts()
    grp_pct = (grp_counts / grp_counts.sum() * 100)

    dest_per_user = destinations.groupby([id_col, group_col]).size().reset_index(name='n_dest')

    stats = dest_per_user.groupby(group_col)['n_dest'].agg(['mean', 'median'])

    groups = grp_pct.index.tolist()
    x = np.arange(len(groups))

    fig, ax1 = plt.subplots(figsize=(12, 7))
    bars = ax1.bar(x, grp_pct.values, color='#ffb74d', edgecolor='#e65100', alpha=0.8, label='% of evacuees')
    ax1.set_ylabel('Percentage (%)', fontsize=13, color='#e65100')

    ax2 = ax1.twinx()
    means = [stats.loc[g, 'mean'] if g in stats.index else 0 for g in groups]
    medians = [stats.loc[g, 'median'] if g in stats.index else 0 for g in groups]
    ax2.plot(x, means, 'o-', color='#1565c0', linewidth=2, markersize=8, label='Mean # destinations')
    ax2.plot(x, medians, 's--', color='#4fc3f7', linewidth=2, markersize=8, label='Median # destinations')
    ax2.set_ylabel('Number of Destinations', fontsize=13, color='#1565c0')

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=20, ha='right', fontsize=11)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    ax1.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_destination_count(
    destinations: pd.DataFrame,
    *,
    id_col: str = "ID",
    title: str = "Distribution of Number of Destinations per Evacuee",
    output_path: str = None,
):
    """
    Bar chart showing how many evacuees had 1, 2, 3, … destinations.
    """
    import matplotlib.pyplot as plt

    counts = destinations.groupby(id_col).size().value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(counts.index, counts.values, edgecolor='black', color='#7986cb')
    ax.set_xlabel('Number of Destinations', fontsize=13)
    ax.set_ylabel('Frequency (Evacuees)', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


# ==========================================================================
# ADVANCED DESTINATION ANALYTICS
# ==========================================================================

def ks_test_distances(
    destinations: pd.DataFrame,
    *,
    group_col: str = "Category",
    distance_col: str = "eu_distance_km",
    plot: bool = True,
    title: str = "K-S Test: Pairwise Distance Distributions",
    output_path: str = None,
) -> pd.DataFrame:
    """
    Kolmogorov-Smirnov tests comparing travel-distance distributions
    between all pairs of evacuee groups.

    Returns a summary DataFrame of pairwise KS statistics and p-values,
    and optionally plots a matrix heatmap.

    Parameters
    ----------
    destinations : DataFrame
        Must contain ``group_col`` and ``distance_col``.
    group_col : str
        Column identifying the evacuee group.
    distance_col : str
        Column with distance values (km).
    plot : bool
        If True, render an annotated heatmap of KS statistics.
    """
    from scipy.stats import ks_2samp
    import matplotlib.pyplot as plt

    groups = sorted(destinations[group_col].dropna().unique())
    n = len(groups)
    ks_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))
    results = []

    for i in range(n):
        for j in range(i + 1, n):
            d1 = destinations.loc[destinations[group_col] == groups[i], distance_col].dropna()
            d2 = destinations.loc[destinations[group_col] == groups[j], distance_col].dropna()
            if len(d1) < 2 or len(d2) < 2:
                continue
            stat, pval = ks_2samp(d1, d2)
            ks_matrix[i, j] = stat
            ks_matrix[j, i] = stat
            p_matrix[i, j] = pval
            p_matrix[j, i] = pval
            results.append({
                'Group_A': groups[i],
                'Group_B': groups[j],
                'KS_Statistic': round(stat, 4),
                'p_value': round(pval, 6),
                'Significant (p<0.05)': pval < 0.05,
                'n_A': len(d1),
                'n_B': len(d2),
            })

    results_df = pd.DataFrame(results)

    if plot and n > 1:
        fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
        im = ax.imshow(ks_matrix, cmap='YlOrRd', vmin=0, vmax=max(0.3, ks_matrix.max()))

        ax.set_xticks(range(n))
        ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=10)
        ax.set_yticks(range(n))
        ax.set_yticklabels(groups, fontsize=10)

        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, '—', ha='center', va='center', fontsize=10)
                else:
                    sig = '***' if p_matrix[i, j] < 0.001 else ('**' if p_matrix[i, j] < 0.01 else ('*' if p_matrix[i, j] < 0.05 else ''))
                    color = 'white' if ks_matrix[i, j] > ks_matrix.max() * 0.5 else 'black'
                    ax.text(j, i, f'{ks_matrix[i,j]:.3f}{sig}', ha='center', va='center', fontsize=9, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label='KS Statistic')
        ax.set_title(f'{title}\n(* p<0.05, ** p<0.01, *** p<0.001)', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    return results_df


def plot_temporal_destinations(
    destinations: pd.DataFrame,
    *,
    date_col: str = "dest_date",
    distance_col: str = "eu_distance_km",
    group_col: str | None = None,
    title: str = "Destination Distance Over Time",
    output_path: str = None,
):
    """
    Track how evacuation destinations evolve over the fire timeline.

    Plots daily median distance to destination with a scatter of individual
    points, showing whether evacuees move further away or closer to home
    as the fire progresses.
    """
    import matplotlib.pyplot as plt

    df = destinations.copy()
    df['_date'] = pd.to_datetime(df[date_col])

    fig, ax = plt.subplots(figsize=(14, 7))

    if group_col and group_col in df.columns:
        colors = plt.cm.Set2.colors
        for idx, (g, grp) in enumerate(sorted(df.groupby(group_col))):
            c = colors[idx % len(colors)]
            daily = grp.groupby('_date')[distance_col].agg(['median', 'count', 'mean'])
            ax.scatter(grp['_date'], grp[distance_col], alpha=0.15, s=8, color=c)
            ax.plot(daily.index, daily['median'], 'o-', color=c, linewidth=2, markersize=5, label=f'{g} (median)')
    else:
        daily = df.groupby('_date')[distance_col].agg(['median', 'count', 'mean'])
        ax.scatter(df['_date'], df[distance_col], alpha=0.15, s=8, color='#7986cb')
        ax.plot(daily.index, daily['median'], 'o-', color='#1a237e', linewidth=2.5, markersize=6, label='Median distance')

    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Euclidean Distance from Home (km)', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_title(title, fontsize=15, fontweight='bold')

    # Add daily destination count as annotation bar
    daily_all = df.groupby('_date')[distance_col].count()
    ax2 = ax.twinx()
    ax2.bar(daily_all.index, daily_all.values, alpha=0.12, color='gray', width=0.8, label='# destinations')
    ax2.set_ylabel('Number of Destinations', fontsize=11, color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def plot_return_by_destination(
    destinations: pd.DataFrame,
    evac_metrics: pd.DataFrame,
    *,
    id_col: str = "ID",
    type_col: str = "dest_type",
    distance_col: str = "eu_distance_km",
    return_col: str = "ReturnDate",
    departure_col: str = "DepartureDate",
    title: str = "Return Behavior by Destination Type",
    output_path: str = None,
):
    """
    Analyze and plot the relationship between destination type/distance
    and when evacuees return home.

    Creates a two-panel figure:
    - Left: box plot of trip duration (days away) by destination type
    - Right: scatter of distance vs. days away, colored by destination type
    """
    import matplotlib.pyplot as plt

    # Merge return info onto primary (first) destination per evacuee
    first_dest = destinations.sort_values('dest_order').drop_duplicates(subset=[id_col], keep='first')
    merged = first_dest.merge(
        evac_metrics[[id_col, return_col, departure_col]].drop_duplicates(subset=[id_col]),
        on=id_col, how='inner',
    )

    merged[return_col] = pd.to_datetime(merged[return_col], errors='coerce')
    merged[departure_col] = pd.to_datetime(merged[departure_col], errors='coerce')
    merged['days_away'] = (merged[return_col] - merged[departure_col]).dt.total_seconds() / 86400
    merged = merged.dropna(subset=['days_away'])
    merged = merged[merged['days_away'] > 0]

    if merged.empty:
        print("No valid return data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # -- Panel 1: Days away by destination type --
    if type_col in merged.columns and merged[type_col].notna().any():
        types = sorted(merged[type_col].dropna().unique())
        data_by_type = [merged.loc[merged[type_col] == t, 'days_away'].values for t in types]
        bp = ax1.boxplot(data_by_type, labels=types, patch_artist=True, showfliers=False)
        colors_map = {
            'residential': '#4fc3f7', 'hotel/motel': '#ff8a65',
            'commercial': '#81c784', 'public': '#ba68c8',
            'road': '#ffb74d', 'other': '#90a4ae',
        }
        for patch, t in zip(bp['boxes'], types):
            patch.set_facecolor(colors_map.get(t, '#bdbdbd'))
        ax1.set_ylabel('Days Away', fontsize=12)
        ax1.set_title('Trip Duration by Destination Type', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='x', rotation=30)
    else:
        ax1.hist(merged['days_away'], bins=20, edgecolor='black', color='#7986cb')
        ax1.set_xlabel('Days Away')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Trip Duration', fontsize=13, fontweight='bold')

    # -- Panel 2: Distance vs days away --
    if type_col in merged.columns and merged[type_col].notna().any():
        for t in sorted(merged[type_col].dropna().unique()):
            sub = merged[merged[type_col] == t]
            c = colors_map.get(t, '#bdbdbd')
            ax2.scatter(sub[distance_col], sub['days_away'], alpha=0.5, s=20, color=c, label=t)
        ax2.legend(fontsize=10, loc='upper right')
    else:
        ax2.scatter(merged[distance_col], merged['days_away'], alpha=0.4, s=15, color='#7986cb')

    ax2.set_xlabel('Euclidean Distance from Home (km)', fontsize=12)
    ax2.set_ylabel('Days Away', fontsize=12)
    ax2.set_title('Distance vs. Trip Duration', fontsize=13, fontweight='bold')

    fig.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
