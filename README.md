# EMBER: Evacuation Mobility Behavior Extraction and Routing

`ember` is a highly modular Python package for inferring wildfire evacuation behavior from GPS-derived human mobility data. 

It is designed to consume pre-processed mobility data (either raw GPS pings or cleaned stay-points) and apply standardized spatiotemporal algorithms to classify behavior according to the 7-category taxonomy (Nima et al., 2025; Zhao et al., 2022).

## Installation

As EMBER is currently in development and not yet available on PyPI, you should install it from source by cloning the repository:

```bash
git clone <repository_url>
cd ember
pip install -e .
```

Requires Python 3.9+ and depends heavily on `pandas`, `geopandas`, and `shapely`.

## Core Modules

EMBER is designed to abstract away the boilerplate CRS transformations and spatial joins, leaving you with simple mathematical/logical pipelines:

*   **`ember.io`**: Flexible loaders for raw shard pickles, pre-computed `evacuation_metrics.csv`, and `home.csv`. Handled CRS normalization automatically.
*   **`grid_based_home_detection`**: The external PyPI package used to infer home locations from nighttime GPS pings.
*   **`ember.zones`**: Matches assigned home locations to timestamped spatial evacuation polygons (Order, Warning, Buffer definitions).
*   **`ember.behavior`**: The core taxonomy classifier mapping 7 distinct behavioral outcomes (SELE, FEUO, FEUW, PERE, SEFN, NER, UR) based on departure timing and zone constraints.
*   **`ember.departure`**: Infers $T_{dep}$ and $T_{ret}$ from raw GPS sequences by detecting extended trips away from the proxy home.
*   **`ember.metrics`**: Generates high-level aggregated metrics and publication-ready visualizations, including Evacuation Compliance Rates, cumulative departure curves, temporal heatmaps, return behavior timelines, spatial delay maps, and the micro-macro DEDI (Damage-Evacuation Disparity Index).

## Getting Started

Check out the interactive `examples/la_fire_example.ipynb` notebook to see the full pipeline in action. It demonstrates how to initialize the data, execute spatial joins, apply the 7-category taxonomy trees, and generate visualizations.

```python
import ember
import pandas as pd

# 1. Load data (handles EPSG:4326 standardization)
homes = ember.io.load_homes("data/home.csv")
zones = ember.io.load_fire_timeline("data/Fire_Timeline/")

# 2. Classify zones spatially
homes_with_zones = ember.zones.classify_zones(
    homes, zones, 
    buffer_distance=2000.0, 
    order_status="Evacuation Order",
    warning_status="Evacuation Warning"
)

# 3. Load pre-computed evacuation stops and metrics
metrics = ember.io.load_evacuation_metrics("data/evacuation_metrics.csv")

# 4. Classify behavior taxonomy
metrics['Category'] = ember.behavior.classify(
    metrics,
    zone_col='ZoneType',
    departure_col='DepartureDate',
    order_start_col='OrderStart'
)

# 5. Print Summary & Plot Visualizations
print(ember.behavior.summary(metrics['Category']))

ember.metrics.plot_departure_curve(
    df=metrics, 
    start_ref=pd.Timestamp('2025-01-07 10:30:00'), 
    group_col='ZoneType', 
    title="Cumulative Departure by Zone"
)

ember.metrics.plot_evacuation_composition(
    metrics, group_col='FireEvent', dual_panel=True, 
    title="Evacuation Composition Breakdown"
)
```

## Running Tests

EMBER includes a `pytest` suite ensuring all core algorithms boundary-match accurately:

```bash
pytest ember/tests/ -v
```

This tests zone buffering and departure logic.

## References

1. Janfeshanaraghi, N., Sun, Y., Zhao, X., Singh, D., Moridpour, S., Bénichou, N., & Kuligowski, E. (2025). Generalized algorithm for inferring wildfire evacuation decisions using large-scale mobile location data. *SSRN*. https://doi.org/10.2139/ssrn.5854973
2. Zhao, X., Xu, Y., Lovreglio, R., et al. (2022). Estimating Wildfire Evacuation Decision and Departure Timing Using Large-Scale GPS Data. *Transportation Research Part D: Transport and Environment*, 107, 103277. https://doi.org/10.1016/j.trd.2022.103277
3. Recalde, A., Sameen, M., Zhang, X., & Zhao, X. (2025). GHOST: Grid-based Home detection via Stay-Time. *GitHub*. https://github.com/SERMOS-LAB/Grid-Based-Home-Detection
