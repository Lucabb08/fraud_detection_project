"""Train and evaluate a VotingClassifier (soft voting) using logistic + rf."""

from src.data_loader import load_and_split, get_num_cat_columns
from src.models import build_logistic_pipeline, build_random_forest, build_voting
from src.evaluation import print_scores
import joblib


def main():
    X_train, X_test, y_train, y_test = load_and_split('data/raw/creditcard.csv')
    num_cols, cat_cols = get_num_cat_columns(X_train)

    lr = build_logistic_pipeline(num_cols=num_cols, cat_cols=cat_cols)
    rf = build_random_forest()

    # Fit base models first
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    vc = build_voting(estimators=[('lr', lr), ('rf', rf)], voting='soft')
    # Fit the voting classifier (it will refit underlying estimators)
    vc.fit(X_train, y_train)

    y_proba = vc.predict_proba(X_test)[:, 1]
    y_pred = vc.predict(X_test)

    print_scores(y_test, y_pred, y_proba)
    joblib.dump(vc, 'results/voting.joblib')


if __name__ == '__main__':
    main()
