"""Reproducible train/CV/test regression benchmark and reporting."""
import json
from pathlib import Path
import argparse
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .load_data import load_dataset, FEATURES, TARGET
from .preprocess import annotate_strength, quality_report, validate_data

SEED = 42


def models():
    return {
        'linear_regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'random_forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                            random_state=SEED, n_jobs=1))
        ])
    }


def evaluate(y_true, predicted):
    return {
        'mae_mpa': round(float(mean_absolute_error(y_true, predicted)), 4),
        'rmse_mpa': round(float(np.sqrt(mean_squared_error(y_true, predicted))), 4),
        'r2': round(float(r2_score(y_true, predicted)), 4),
    }


def run(df, output=Path('reports')):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    df = validate_data(df)
    if len(df) < 30:
        raise ValueError('Need at least 30 observations for this 5-fold benchmark')
    report = quality_report(df)
    df = annotate_strength(df)
    # Annotation is for visualization ONLY; no strength-derived labels fed into ML.
    df.to_csv(output / 'annotated_exploratory.csv', index=False)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = models()
    cv_results = {}
    for name, model in candidates.items():
        scores = -cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error',
                                  cv=cv, n_jobs=1)
        cv_results[name] = {'cv_mae_mean_mpa': round(float(scores.mean()), 4),
                            'cv_mae_std_mpa': round(float(scores.std()), 4)}
    winner = min(cv_results, key=lambda key: cv_results[key]['cv_mae_mean_mpa'])
    trained = candidates[winner].fit(X_train, y_train)
    pred = trained.predict(X_test)
    test_scores = evaluate(y_test, pred)
    result = {
        'dataset_quality': report, 'train_rows': len(X_train), 'test_rows': len(X_test),
        'random_state': SEED, 'model_selection': 'lowest training 5-fold CV MAE',
        'cross_validation': cv_results, 'selected_model': winner,
        'held_out_test': test_scores,
        'notes': ['Holdout test is evaluated once after model choice by training-only CV.',
                  'Exploratory labels are NOT predictors or engineering strength grades.',
                  'Duplicate records are reported, not silently deleted.']
    }
    (output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    joblib.dump(trained, output / 'selected_model.joblib')
    predictions = X_test.copy()
    predictions['actual_strength_mpa'] = y_test.to_numpy()
    predictions['predicted_strength_mpa'] = pred
    predictions.to_csv(output / 'test_predictions.csv', index=False)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, pred, alpha=0.55, s=18)
    vmin = min(float(y_test.min()), float(pred.min()))
    vmax = max(float(y_test.max()), float(pred.max()))
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', label='Perfect prediction')
    plt.xlabel('Measured strength (MPa)')
    plt.ylabel('Predicted strength (MPa)')
    plt.title(f'Holdout predictions: {winner}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / 'actual_vs_predicted.png', dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(df[TARGET], bins=25)
    plt.xlabel('Measured strength (MPa)')
    plt.ylabel('Number of samples')
    plt.title('Concrete strength distribution')
    plt.tight_layout()
    plt.savefig(output / 'strength_distribution.png', dpi=150)
    plt.close()

    summary = f'''# Model results (generated)\n\nRows: {len(df)}. Train: {len(X_train)}. Held-out test: {len(X_test)}.\n\nModel selection used 5-fold cross-validation **on training data only**, minimizing MAE.\n\n| Model | CV MAE mean (MPa) | CV MAE std (MPa) |\n|---|---:|---:|\n'''
    for name, score in cv_results.items():
        summary += f"| {name} | {score['cv_mae_mean_mpa']:.4f} | {score['cv_mae_std_mpa']:.4f} |\n"
    summary += f"\nSelected: **{winner}**.\n\nHeld-out MAE: **{test_scores['mae_mpa']:.4f} MPa**, RMSE: **{test_scores['rmse_mpa']:.4f} MPa**, R²: **{test_scores['r2']:.4f}**.\n\nExploratory annotations are relative dataset tertiles; they are not structural safety ratings. The model is experimental and must not be used for engineering design.\n"
    (output / 'results.md').write_text(summary, encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('data/concrete.csv'))
    parser.add_argument('--output', type=Path, default=Path('reports'))
    args = parser.parse_args()
    df = load_dataset(path=args.dataset)
    results = run(df, output=args.output)
    print(json.dumps(results, indent=2))
