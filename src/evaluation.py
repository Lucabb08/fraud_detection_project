"""Evaluation and visualization helpers."""

from typing import Optional, Dict
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import json


def print_scores(y_true, y_pred, y_proba: Optional[float] = None):
    print(f"precision score : {precision_score(y_true,y_pred)}")
    print(f"recall score : {recall_score(y_true,y_pred)}")
    print(f"F1 score : {f1_score(y_true,y_pred)}")
    if y_proba is not None:
        print("AUPRC :", average_precision_score(y_true, y_proba))


def save_metrics(path: str, y_true, y_pred, y_proba: Optional[float] = None) -> None:
    metrics: Dict[str, float] = {
        'precision': float(precision_score(y_true, y_pred)),
        'recall': float(recall_score(y_true, y_pred)),
        'f1': float(f1_score(y_true, y_pred)),
    }
    if y_proba is not None:
        metrics['auprc'] = float(average_precision_score(y_true, y_proba))

    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
