"""Source package for the project.

Top-level modules:
- `src.data_loader` : functions to load the dataset and perform the chronological split (use `load_and_split`).
- `src.models`      : builder functions that return sklearn estimators or pipelines (e.g. `build_logistic_pipeline`).
- `src.evaluation`  : helpers to print and save evaluation metrics.

Usage example:
    from src.data_loader import load_and_split
    X_train, X_test, y_train, y_test = load_and_split('data/raw/creditcard.csv')

Keep experiments under `src/experiments/` and run them with `python -m src.experiments.train_logistic` or `python -m src.train`.
"""

__all__ = ["data_loader", "models", "evaluation"]
