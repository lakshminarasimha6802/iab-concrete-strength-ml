"""Acquire and cache the UCI Concrete Compressive Strength dataset (ID 165)."""
from pathlib import Path
import argparse
import pandas as pd

DATA_FILE = Path('data/concrete.csv')
FEATURES = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer',
            'coarse_aggregate', 'fine_aggregate', 'age_days']
TARGET = 'strength_mpa'


def load_dataset(path=DATA_FILE, refresh=False):
    """Load cached CSV or retrieve official data via ucimlrepo (internet needed once)."""
    path = Path(path)
    if path.exists() and not refresh:
        return pd.read_csv(path)
    try:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=165)
    except Exception as exc:
        raise RuntimeError('Could not download UCI dataset 165. Connect to the internet, '
                           'install requirements, and rerun; cached CSV also works offline.') from exc
    X, y = dataset.data.features, dataset.data.targets
    if X is None or y is None or X.shape[1] != 8 or y.shape[1] != 1:
        raise ValueError('Unexpected UCI dataset shape; review upstream data before training.')
    result = X.copy()
    result.columns = FEATURES  # UCI's documented order; avoid dependency on long labels
    result[TARGET] = y.iloc[:, 0].to_numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    df = load_dataset(refresh=args.refresh)
    print(f'Saved/loaded {df.shape[0]} observations with {df.shape[1] - 1} features at {DATA_FILE}')
