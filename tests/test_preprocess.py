import numpy as np
import pandas as pd
import pytest
from src.load_data import FEATURES, TARGET
from src.preprocess import annotate_strength, quality_report, validate_data
from src.train_model import evaluate, run


def sample_df(rows=60):
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.uniform(1, 100, (rows, 8)), columns=FEATURES)
    df['age_days'] = rng.integers(1, 365, rows)
    df[TARGET] = df['cement'] * .2 + df['age_days'] * .03 + rng.normal(0, 2, rows) + 20
    return df


def test_valid():
    data = sample_df()
    assert validate_data(data).shape == (60, 9)
    assert quality_report(data)['rows'] == 60


def test_reject_missing_column():
    with pytest.raises(ValueError, match='Missing expected'):
        validate_data(sample_df().drop(columns=['cement']))


def test_reject_negative_amount():
    data = sample_df()
    data.loc[0, 'water'] = -10
    with pytest.raises(ValueError, match='negative'):
        validate_data(data)


def test_reject_missing_values():
    data = sample_df()
    data.loc[0, 'cement'] = np.nan
    with pytest.raises(ValueError, match='Missing or invalid'):
        validate_data(data)


def test_annotations_not_features():
    annotated = annotate_strength(sample_df())
    assert set(annotated['strength_band_exploratory']) == {'lower', 'middle', 'upper'}
    assert 'strength_band_exploratory' not in FEATURES


def test_metrics_perfect_prediction():
    metrics = evaluate([10, 20, 30], [10, 20, 30])
    assert metrics['mae_mpa'] == 0
    assert metrics['r2'] == 1


def test_training_smoke(tmp_path):
    results = run(sample_df(), output=tmp_path)
    assert results['selected_model'] in ('linear_regression', 'random_forest')
    for filename in ['metrics.json', 'results.md', 'selected_model.joblib',
                     'actual_vs_predicted.png', 'test_predictions.csv']:
        assert (tmp_path / filename).exists()
