# Outputs

After running `python -m src.train_model`, this folder contains:
- `metrics.json`: source-data diagnostics, five-fold training CV MAEs, and held-out metrics
- `results.md`: generated short report
- `test_predictions.csv`: holdout observations and predictions
- `annotated_exploratory.csv`: research-only relative strength bands
- `strength_distribution.png`, `actual_vs_predicted.png`: graphs
- `selected_model.joblib`: trained model (load only from trusted sources)

Generated files are excluded from Git by default. If publishing results, review them first and intentionally commit your report/plots as appropriate.
