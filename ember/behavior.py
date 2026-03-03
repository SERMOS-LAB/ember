"""
Evacuation behavior taxonomy classifier.
"""

import pandas as pd
from typing import Optional, Dict

from ._constants import CATEGORY_GROUP_IDS, CATEGORY_NAMES


def classify(
    records: pd.DataFrame,
    *,
    zone_col: str = "ZoneType",
    departure_col: str = "DepartureDate",
    return_col: str = "ReturnDate",
    order_start_col: str = "OrderStart",
    order_end_col: Optional[str] = "OrderEnd",
    fire_start_col: Optional[str] = None,
) -> pd.Series:
    """
    Classify evacuation behavior into the 7-category taxonomy.
    
    Categories: SELE, SEFN, FEUO, FEUW, PERE, NER, UR.
    
    Parameters
    ----------
    records : pd.DataFrame
        DataFrame with evacuation metrics.
    zone_col : str
        Column name for ZoneType ('Order', 'Warning', 'Buffer', 'Outside').
    departure_col : str
        Column name for departure timestamp.
    return_col : str
        Column name for return timestamp.
    order_start_col : str
        Column name for when the order/warning was issued.
    fire_start_col : str, optional
        Column name for when the fire started. If None, falls back to order_start.
        
    Returns
    -------
    pd.Series
        A pandas Series of string category codes ('SELE', etc), with the same index as records.
    """
    
    # We will build a logic string based on conditions, 
    # to efficiently vectorize using numpy select or pandas apply.
    # Since conditions can be complex with NaNs, apply over rows is usually safe enough 
    # for the scale of evacuation metrics (typically N < 1,000,000).
    
    def _classify_row(row):
        dep_time = row.get(departure_col)
        ret_time = row.get(return_col)
        order_start = row.get(order_start_col)
        zone = row.get(zone_col, 'Unknown')
        
        fire_start = row.get(fire_start_col) if fire_start_col else None
        order_end = row.get(order_end_col) if order_end_col else None
        
        # Missing order start? We can't classify compliant vs non-compliant accurately
        if pd.isna(order_start):
            return 'UR'
            
        # No departure -> NER
        if pd.isna(dep_time):
            return 'NER'
            
        # --- Pre-order departure ---
        if dep_time < order_start:
            # Returned before order -> not an evacuation trip
            if not pd.isna(ret_time) and ret_time < order_start:
                return 'NER'
                
            # Departed before fire started -> Trip, not evacuation
            if fire_start is not None and not pd.isna(fire_start):
                if dep_time < fire_start:
                    return 'NER'
                
            # Self-Evacuee
            return 'SELE'
            
        # --- Post-order departure ---
        if zone == 'Buffer':
            return 'SEFN'
            
        # If order_end is provided, we can classify PERE (Partially Compliant / Return Early)
        # People who returned before the order was lifted
        if order_end is not None and not pd.isna(order_end):
            if not pd.isna(ret_time) and ret_time <= order_end:
                return 'PERE'
                
        if zone == 'Warning':
            return 'FEUW'
            
        if zone == 'Order':
            return 'FEUO'
            
        # If 'Outside' or unknown and departing
        return 'UR'

    result = records.apply(_classify_row, axis=1)
    
    # Cast to Categorical
    cat_type = pd.CategoricalDtype(
        categories=['SELE', 'PERE', 'FEUO', 'FEUW', 'SEFN', 'NER', 'UR'], 
        ordered=True
    )
    return result.astype(cat_type)


def summary(classified: pd.Series) -> pd.DataFrame:
    """
    Generate a summary table of behavior categories.
    """
    counts = classified.value_counts(dropna=False)
    pct = (counts / len(classified) * 100).round(2)
    
    df = pd.DataFrame({
        'Count': counts,
        'Percentage': pct
    }).reset_index()
    df.columns = ['Category', 'Count', 'Percentage']
    
    # Map to full names
    df['Description'] = df['Category'].map(CATEGORY_NAMES)
    df['GroupId'] = df['Category'].map(CATEGORY_GROUP_IDS)
    
    return df.sort_values('GroupId').drop(columns=['GroupId'])
