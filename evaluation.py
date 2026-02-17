from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import numpy as np


def evaluate(y_true, y_pred, y_prob):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    recall = recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    f1 = f1_score(
        y_true, y_pred, average="weighted", zero_division=0
    )

    try:
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(
                y_true, y_prob,
                multi_class="ovr",
                average="weighted"
            )
    except:
        auc = 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4)
    }
