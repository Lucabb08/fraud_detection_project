"""Train multiple models and save them to `results/`.

Usage: python -m src.train
"""
from src.data_loader import load_and_split, get_num_cat_columns
from src.models import get_default_builders, build_voting, build_logistic_pipeline, build_random_forest
from src.evaluation import print_scores
import joblib


def main():
    X_train, X_test, y_train, y_test = load_and_split('data/raw/creditcard.csv')
    num_cols, cat_cols = get_num_cat_columns(X_train)

    builders = get_default_builders()

    for name, builder in builders.items():
        print(f"Training {name} ...")
        # For pipelines that accept num/cat cols we pass them
        try:
            mdl = builder(num_cols=num_cols, cat_cols=cat_cols)
        except TypeError:
            mdl = builder()

        mdl.fit(X_train, y_train)

        y_pred = mdl.predict(X_test)
        y_proba = mdl.predict_proba(X_test)[:, 1] if hasattr(mdl, 'predict_proba') else None
        print(f"Results for {name}:")
        print_scores(y_test, y_pred, y_proba)
        joblib.dump(mdl, f'results/{name}.joblib')

    # Example: train a voting ensemble using logistic + rf
    print("Training voting ensemble ...")
    lr = build_logistic_pipeline(num_cols=num_cols, cat_cols=cat_cols)
    rf = build_random_forest()
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    vc = build_voting(estimators=[('lr', lr), ('rf', rf)], voting='soft')
    # Fit the voting classifier (it will refit underlying estimators)
    vc.fit(X_train, y_train)
    y_pred = vc.predict(X_test)
    y_proba = vc.predict_proba(X_test)[:, 1]
    print_scores(y_test, y_pred, y_proba)
    joblib.dump(vc, 'results/voting.joblib')


if __name__ == '__main__':
    main()
