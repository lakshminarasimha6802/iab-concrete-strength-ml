# Data source

Dataset: Concrete Compressive Strength by I-Cheng Yeh (1998), UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C5PK67
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/
Original dataset has 1,030 records, 8 feature columns and a single strength target (MPa).

Run `python -m src.load_data` from the repository root to create `data/concrete.csv`.
The dataset is not redistributed in Git; the script fetches it from UCI on first run.
The features are renamed by their *documented original column order*. Should the upstream
schema change, the script will halt on column-count mismatch, and the mapping should be reviewed.
