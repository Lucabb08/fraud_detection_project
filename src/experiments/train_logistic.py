"""Train and evaluate a Logistic Regression pipeline."""

from src.data_loader import load_and_split, get_num_cat_columns
from src.models import build_logistic_pipeline
from src.evaluation import print_scores
import joblib


def main():
    X_train, X_test, y_train, y_test = load_and_split('data/raw/creditcard.csv')
    num_cols, cat_cols = get_num_cat_columns(X_train)

    pipe = build_logistic_pipeline(num_cols=num_cols, cat_cols=cat_cols)
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba > 0.5).astype(int)

    print_scores(y_test, y_pred, y_proba)
    joblib.dump(pipe, 'results/logistic_pipeline.joblib')


if __name__ == '__main__':
    main()
