"""Train and evaluate a Bagging classifier with a decision tree base."""

from src.data_loader import load_and_split
from src.models import build_bagging, build_tree
from src.evaluation import print_scores
import joblib


def main():
    X_train, X_test, y_train, y_test = load_and_split('data/raw/creditcard.csv')

    base = build_tree()
    bc = build_bagging(base_estimator=base, n_estimators=10)
    bc.fit(X_train, y_train)

    y_pred = bc.predict(X_test)
    y_proba = bc.predict_proba(X_test)[:, 1]

    print_scores(y_test, y_pred, y_proba)
    joblib.dump(bc, 'results/bagging.joblib')


if __name__ == '__main__':
    main()
