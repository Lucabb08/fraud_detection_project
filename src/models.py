"""Model builders returning sklearn estimators or pipelines.

Provide functions like `build_logistic_pipeline`, `build_knn_pipeline`, `build_tree`, etc.
"""

from typing import List, Optional, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, VotingClassifier


def _make_num_cat_preprocessor(num_cols: Optional[List[str]] = None,
                               cat_cols: Optional[List[str]] = None):
    """Build a ColumnTransformer for numeric and categorical columns.

    If num_cols or cat_cols is None or empty, returns None (no preprocessor).
    """
    transformers = []
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols))
    if transformers:
        return ColumnTransformer(transformers, remainder='drop')
    return None


def build_logistic_pipeline(num_cols: Optional[List[str]] = None,
                            cat_cols: Optional[List[str]] = None,
                            **kwargs) -> Pipeline:
    """Return a pipeline with optional preprocessing and LogisticRegression."""
    preproc = _make_num_cat_preprocessor(num_cols, cat_cols)
    steps = []
    if preproc:
        steps.append(('preproc', preproc))
    steps.append(('clf', LogisticRegression(class_weight='balanced', max_iter=1000, **kwargs)))
    return Pipeline(steps)


def build_knn_pipeline(num_cols: Optional[List[str]] = None,
                       cat_cols: Optional[List[str]] = None,
                       n_neighbors: int = 5,
                       weights: str = 'distance',
                       **kwargs) -> Pipeline:
    """Pipeline for KNN (needs scaling)."""
    preproc = _make_num_cat_preprocessor(num_cols, cat_cols)
    steps = []
    if preproc:
        steps.append(('preproc', preproc))
    steps.append(('clf', KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, **kwargs)))
    return Pipeline(steps)


def build_tree(class_weight: str = 'balanced', **kwargs) -> DecisionTreeClassifier:
    """Return a DecisionTreeClassifier (no scaling needed)."""
    return DecisionTreeClassifier(criterion='entropy', class_weight=class_weight, **kwargs)


def build_random_forest(n_estimators: int = 200, class_weight: str = 'balanced', random_state: int = 42, **kwargs) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n_estimators, class_weight=class_weight, random_state=random_state, **kwargs)


def build_adaboost(base_estimator=None, n_estimators: int = 50, **kwargs) -> AdaBoostClassifier:
    """Return an AdaBoostClassifier. Try to pass the base estimator using the
    newer 'estimator' keyword first (scikit-learn >= 1.2), otherwise fall back
    to 'base_estimator' for older versions."""
    try:
        return AdaBoostClassifier(estimator=base_estimator, n_estimators=n_estimators, **kwargs)
    except TypeError:
        return AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, **kwargs)


def build_bagging(base_estimator=None, n_estimators: int = 10, **kwargs) -> BaggingClassifier:
    """Return a BaggingClassifier. Accept both 'estimator' and 'base_estimator'."""
    try:
        return BaggingClassifier(estimator=base_estimator, n_estimators=n_estimators, **kwargs)
    except TypeError:
        return BaggingClassifier(base_estimator=base_estimator, n_estimators=n_estimators, **kwargs)


def build_voting(estimators: List = None, voting: str = 'soft', weights: Optional[List[float]] = None) -> VotingClassifier:
    """Build a voting ensemble. estimators is a list of tuples: [('lr', lr_pipe), ('rf', rf), ...]"""
    if estimators is None:
        raise ValueError("Provide estimators list, e.g. [('lr', pipe), ('rf', rf)]")
    return VotingClassifier(estimators=estimators, voting=voting, weights=weights)


def get_default_builders() -> Dict[str, Any]:
    """Return a dict of name -> builder functions to iterate/train easily."""
    return {
        'logistic': build_logistic_pipeline,
        'knn': build_knn_pipeline,
        'tree': build_tree,
        'random_forest': build_random_forest,
        'adaboost': build_adaboost,
        'bagging': build_bagging,
    }


if __name__ == "__main__":
    print("Example: from src.models import build_logistic_pipeline; pipe = build_logistic_pipeline(num_cols=['V1','V2'])")