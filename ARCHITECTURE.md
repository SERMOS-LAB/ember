# EMBER Package Architecture

The `ember` package is a focused, modular library for inferring wildfire evacuation behavior from GPS data. It abstracts the complex spatiotemporal math, taxonomy trees, and gap metrics into clean Python functions.

## High-Level Pipeline

```mermaid
graph TD
    classDef io fill:#e3f2fd,stroke:#1e88e5,stroke-width:2px,color:#000
    classDef compute fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000
    classDef output fill:#e8f5e9,stroke:#43a047,stroke-width:2px,color:#000
    
    A[Raw GPS Pings]:::io --> B
    B[grid_based_home_detection]:::compute --> C[Homes DataFrame<br>EPSG:4326]:::output
    
    D[Evacuation Zones<br>GeoJSONs]:::io --> E
    C --> E[zones.classify_zones]:::compute
    E --> F[Homes with ZoneType<br>Order/Warning/Buffer/Outside]:::output
    
    A --> G[departure.infer]:::compute
    C --> G
    G --> H[Departure &amp; Return<br>Times]:::output
    
    F --> I
    H --> I[behavior.classify]:::compute
    I --> J[7-Category Taxonomy<br>SELE, FEUO, PERE, etc.]:::output
    
    J --> K[metrics.compliance_rate<br>metrics.dedi<br>metrics.plot_departure_curve]:::compute
    K --> L[Summary Reports &amp; Plots]:::output
```

## Module Definitions: Inputs & Outputs

### 1. `ember.io`
**Purpose**: Standardizes all spatial and tabular data loading, automatically handling CRS transformations to EPSG:4326 (WGS84).
*   **Inputs**: Paths to `home.csv`, `evacuation_metrics.csv`, MongoDB `.pkl` shards, or directory of `Fire_Timeline` GeoJSONs.
*   **Outputs**: Cleaned `pandas.DataFrame` or `geopandas.GeoDataFrame`.

### 2. `grid_based_home_detection` (External PyPI Package)
**Purpose**: Infers proxy home locations from nighttime GPS pings using a localized grid-snapping approach.
*   **Inputs**: `pings` DataFrame (requires `user_id`, `latitude`, `longitude`, `datetime`).
*   **Outputs**: `gpd.GeoDataFrame` of home locations with `num_nights` and `stay_time`.

### 3. `ember.zones`
**Purpose**: Spatially joins resident homes to dynamic, timestamped fire evacuation perimeters.
*   **Inputs**: `homes` (from `grid_based_home_detection`) and `fire_zones` (from `io.py`).
*   **Outputs**: The `homes` dataframe with appended columns: `ZoneType` (Order, Warning, Buffer, Outside) and `OrderStart`.

### 4. `ember.departure`
**Purpose**: Identifies the timestamps when a resident left their home spatial cluster ($T_{dep}$) and returned ($T_{ret}$).
*   **Inputs**: `pings` DataFrame and `order_start` timestamp.
*   **Outputs**: Tuple `(T_dep, T_ret)` for each user.

### 5. `ember.behavior`
**Purpose**: The core deterministic classifier that maps timing and spatial traits into the 7-category taxonomy.
*   **Inputs**: The merged DataFrame containing `ZoneType`, `DepartureDate`, `ReturnDate`, and `OrderStart`.
*   **Outputs**: A `pd.Series` of Categorical strings (e.g., `'SELE'`, `'FEUO'`, `'NER'`).

### 6. `ember.metrics`
**Purpose**: Generates publication-ready metrics and plots.
*   **Inputs**: The classified records DataFrame.
*   **Outputs**: Aggregated DataFrames (Compliance Rates, DEDI) and matplotlib plots (Departure Curves).

## Configurable Hyperparameters

EMBER is designed to give researchers full control over the physical and temporal thresholds that define an evacuation. Every core function exposes these as keyword arguments. 

Here are the key hyperparameters you can manipulate:

*   **Zone Buffering (`buffer_distance`)**: In `ember.zones.classify_zones`, you can adjust the physical width of the Shadow Evacuation zone (default is `1000.0` meters).
*   **Home Inference Grid (`grid_cell_size`)**: In `ember.ghost.infer_homes`, you can change the snapping resolution (default `50.0` meters).
*   **Residency Thresholds (`min_nights`, `min_stay_time`)**: In `ghost.py`, you can strictly define who counts as a resident vs. a transient visitor (defaults strictly to `14` nights).
*   **Trip Detection Radii (`home_radius`, `away_radius`)**: In `ember.departure.infer`, define what physical distance constitutes "leaving the neighborhood" (default `away_radius=1000.0`).
*   **Nighttime Windows (`nighttime_start`, `nighttime_end`)**: In `ghost.py`, define when a user must be present to count as dwelling at home.

```mermaid
sequenceDiagram
    participant User
    participant IO as ember.io
    participant Ghost as grid_based_home_detection
    participant Zones as ember.zones
    participant Behavior as ember.behavior
    
    User->>IO: load_shards("data_sample/")
    IO-->>User: Raw Pings (DataFrame)
    
    User->>Ghost: infer_homes(pings, min_nights=14)
    Ghost-->>User: Homes (GeoDataFrame)
    
    User->>IO: load_fire_timeline("Fire_Timeline/")
    IO-->>User: Zones (GeoDataFrame)
    
    User->>Zones: classify_zones(homes, zones)
    Zones-->>User: Homes with ZoneType
    
    User->>Behavior: classify(records)
    Behavior-->>User: Taxonomy Category (SELE, NER, etc.)
```
