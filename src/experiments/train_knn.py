"""Train and evaluate a KNN pipeline."""

from src.data_loader import load_and_split, get_num_cat_columns
from src.models import build_knn_pipeline
from src.evaluation import print_scores
import joblib


def main():
    X_train, X_test, y_train, y_test = load_and_split('data/raw/creditcard.csv')
    num_cols, cat_cols = get_num_cat_columns(X_train)

    pipe = build_knn_pipeline(num_cols=num_cols, cat_cols=cat_cols)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print_scores(y_test, y_pred, y_proba)
    joblib.dump(pipe, 'results/knn_pipeline.joblib')


if __name__ == '__main__':
    main()
