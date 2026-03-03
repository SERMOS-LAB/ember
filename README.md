# EMBER: Evacuation Mobility Behavior Inference 

`ember` is a highly modular, pip-installable Python package for inferring wildfire evacuation behavior from GPS-derived human mobility data. 

It is designed to consume pre-processed mobility data (either raw GPS pings or cleaned stay-points) and apply standardized spatiotemporal algorithms to classify behavior according to the 7-category taxonomy (Sun et al., 2024; Zhao et al., 2022).

## Installation

You can install `ember` locally in editable mode alongside its dependencies:

```bash
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
*   **`ember.metrics`**: Generates high-level metrics like Evacuation Compliance Rates, cumulative departure curves, and the micro-macro DEDI (Damage-Evacuation Disparity Index).

## Getting Started

### 1. Generate Sample Data
To test the package, you can generate an anonymized sample from your existing LA Fire outputs:
```bash
python examples/generate_sample_data.py
```
This will create an `examples/data_sample/` directory with a mini-cohort of users and timelines to test against.

### 2. Basic Inference Pipeline

Check out the interactive `examples/Ember_Example.ipynb` notebook to see the full pipeline in action. It demonstrates how to initialize the data, execute spatial joins, apply the 7-category taxonomy trees, and generate publication-ready plots.

```python
import ember
import pandas as pd

# Load data (handles EPSG:4326 standardization)
homes = ember.io.load_homes("data_sample/sample_homes.csv")
zones = ember.io.load_fire_timeline("data_sample/Fire_Timeline_Sample/")

# Classify zones
homes_with_zones = ember.zones.classify_zones(
    homes, zones, buffer_distance=1000.0, order_status="Evacuation Order"
)

# Load stops/metrics
metrics = ember.io.load_evacuation_metrics("data_sample/sample_evacuation_metrics.csv")

# Classify behavior taxonomy
metrics['Category'] = ember.behavior.classify(
    metrics,
    zone_col='ZoneType',
    departure_col='DepartureDate',
    order_start_col='OrderStart'
)

# Print Summary
print(ember.behavior.summary(metrics['Category']))
```

## Running Tests

EMBER includes a comprehensive `pytest` suite ensuring all core algorithms boundary-match accurately:

```bash
pytest ember/tests/ -v
```

This tests zone buffering and departure logic.
