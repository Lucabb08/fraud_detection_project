"""Train and evaluate a Random Forest."""

from src.data_loader import load_and_split
from src.models import build_random_forest
from src.evaluation import print_scores
import joblib


def main():
    X_train, X_test, y_train, y_test = load_and_split('data/raw/creditcard.csv')

    rf = build_random_forest()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print_scores(y_test, y_pred, y_proba)
    joblib.dump(rf, 'results/random_forest.joblib')


if __name__ == '__main__':
    main()
