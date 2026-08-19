from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"

FALSE_NEGATIVE_COST = 1200
FALSE_POSITIVE_COST = 60


def cost_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST


def evaluate():
    model = joblib.load(MODELS_DIR / "churn_model.joblib")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    print("Metrics at threshold 0.5:")
    print(f"Precision: {precision_score(y_test, y_pred_default):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_default):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred_default):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred_default)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix (threshold = 0.5)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()

    cost_default = cost_at_threshold(y_test, y_proba, 0.5)

    thresholds = np.arange(0.01, 1.0, 0.01)
    costs = [cost_at_threshold(y_test, y_proba, t) for t in thresholds]
    best_idx = int(np.argmin(costs))
    best_threshold = thresholds[best_idx]
    cost_best = costs[best_idx]

    y_pred_best = (y_proba >= best_threshold).astype(int)

    print(f"\nCost at threshold 0.5:            {cost_default:,} NIS")
    print(f"Optimal threshold:                {best_threshold:.2f}")
    print(f"Cost at optimal threshold:        {cost_best:,} NIS")
    print(f"Savings vs. threshold 0.5:        {cost_default - cost_best:,} NIS")

    print("\nAt threshold 0.5:")
    print(f"  Precision: {precision_score(y_test, y_pred_default):.4f}, Recall: {recall_score(y_test, y_pred_default):.4f}")
    print(f"At optimal threshold ({best_threshold:.2f}):")
    print(f"  Precision: {precision_score(y_test, y_pred_best):.4f}, Recall: {recall_score(y_test, y_pred_best):.4f}")


if __name__ == "__main__":
    evaluate()
