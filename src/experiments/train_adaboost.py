"""Train and evaluate an AdaBoost classifier with a decision tree base."""

from src.data_loader import load_and_split
from src.models import build_adaboost, build_tree
from src.evaluation import print_scores
import joblib


def main():
    X_train, X_test, y_train, y_test = load_and_split('data/raw/creditcard.csv')

    base = build_tree()
    ab = build_adaboost(base_estimator=base, n_estimators=30)
    ab.fit(X_train, y_train)

    y_pred = ab.predict(X_test)
    y_proba = ab.predict_proba(X_test)[:, 1]

    print_scores(y_test, y_pred, y_proba)
    joblib.dump(ab, 'results/adaboost.joblib')


if __name__ == '__main__':
    main()
