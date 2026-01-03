"""Train and evaluate a Decision Tree."""

from src.data_loader import load_and_split
from src.models import build_tree
from src.evaluation import print_scores
import joblib


def main():
    X_train, X_test, y_train, y_test = load_and_split('data/raw/creditcard.csv')

    tree = build_tree()
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)
    y_proba = tree.predict_proba(X_test)[:, 1]

    print_scores(y_test, y_pred, y_proba)
    joblib.dump(tree, 'results/tree.joblib')


if __name__ == '__main__':
    main()
