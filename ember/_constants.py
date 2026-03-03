"""
Shared constants for the EMBER package.
"""

# =============================================================================
# EVACUATION BEHAVIOR TAXONOMY
# =============================================================================

CATEGORY_NAMES = {
    'SELE': 'Self-Evacuee Leaving Early',
    'SEFN': 'Shadow Evacuee From Nearby',
    'FEUO': 'Fully-compliant Evacuee Under Order',
    'FEUW': 'Fully-compliant Evacuee Under Warning',
    'PERE': 'Partially-compliant Evacuee (Early Returner)',
    'NER':  'Non-Evacuee Resident',
    'UR':   'Uncategorized Resident',
}

CATEGORY_GROUP_IDS = {
    'SELE': 1,
    'PERE': 2,
    'FEUW': 3,
    'FEUO': 4,
    'SEFN': 5,
    'NER':  6,
    'UR':   7,
}

CATEGORY_COLORS = {
    'SELE': '#2196F3',   # Blue 
    'PERE': '#4CAF50',   # Green 
    'FEUW': '#FF9800',   # Orange 
    'FEUO': '#F44336',   # Red 
    'SEFN': '#9C27B0',   # Purple 
    'NER':  '#607D8B',   # Gray 
    'UR':   '#BDBDBD',   # Light gray 
}

# =============================================================================
# ZONES
# =============================================================================

ZONE_TYPES = ['Order', 'Warning', 'Buffer']

ZONE_PRIORITY = {
    'Order':   2,
    'Warning': 1,
    'Buffer':  0,
}

# =============================================================================
# CRS DETECCION
# =============================================================================

EPSG_WGS84 = 4326
EPSG_WEB_MERCATOR = 3857
EPSG_UTM_11N = 26911
