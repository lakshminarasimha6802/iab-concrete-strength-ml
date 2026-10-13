"""Data validation and transparent *exploratory* label creation."""
import numpy as np
import pandas as pd
from .load_data import FEATURES, TARGET


def validate_data(df):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f'Missing expected columns: {missing}')
    df = df[FEATURES + [TARGET]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df.isna().any().any():
        raise ValueError(f'Missing or invalid numerical values: {df.isna().sum().to_dict()}')
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError('Found infinite numerical values')
    if (df[FEATURES].drop(columns='age_days') < 0).any().any():
        raise ValueError('Ingredient quantities cannot be negative')
    if (df['age_days'] <= 0).any() or (df[TARGET] <= 0).any():
        raise ValueError('Age and measured compressive strength must be positive')
    return df


def annotate_strength(df):
    """Exploratory bands based on sample tertiles; NOT standardized structural grades."""
    result = df.copy()
    result['strength_band_exploratory'] = pd.qcut(
        result[TARGET].rank(method='first'), q=3, labels=['lower', 'middle', 'upper']
    ).astype(str)
    return result


def quality_report(df):
    return {
        'rows': int(len(df)),
        'missing_values': {k: int(v) for k, v in df.isna().sum().items()},
        'duplicate_rows': int(df.duplicated().sum()),
        'strength_min_mpa': float(df[TARGET].min()),
        'strength_max_mpa': float(df[TARGET].max()),
    }
