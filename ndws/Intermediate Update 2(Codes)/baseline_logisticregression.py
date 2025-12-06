"""
baseline_logreg.py

Tile-level baseline:
    - Mean-pool each (64x64x12) tile into a 12D feature vector.
    - Label = 1 if any fire pixel exists in the tile, else 0.
    - Train Logistic Regression with class_weight='balanced'.
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
import joblib
import matplotlib.pyplot as plt


def load_data(data_dir: str = "data"):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))  # (N, 64, 64, 12)
    Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))  # (N, 64, 64, 1)
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))

    # Ensure labels are binary
    Y_train = (Y_train > 0).astype(np.uint8)
    Y_val = (Y_val > 0).astype(np.uint8)

    return X_train, Y_train, X_val, Y_val


def build_tile_targets(Y):
    """
    Convert (N, H, W, 1) masks into tile labels:
        1 if any pixel is fire, else 0.
    """
    N = Y.shape[0]
    tile_sum = Y.reshape(N, -1).sum(axis=1)
    return (tile_sum > 0).astype(np.uint8)


def main():
    data_dir = "data"
    outputs_dir = "outputs"
    curves_dir = os.path.join(outputs_dir, "curves")
    models_dir = os.path.join(outputs_dir, "models")

    os.makedirs(curves_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("Loading data...")
    X_train, Y_train, X_val, Y_val = load_data(data_dir)

    # Mean-pool per channel -> (N, 12)
    X_train_feat = X_train.mean(axis=(1, 2))  # 64x64 -> scalar per channel
    X_val_feat = X_val.mean(axis=(1, 2))

    y_train = build_tile_targets(Y_train)
    y_val = build_tile_targets(Y_val)

    print("Train feature shape:", X_train_feat.shape)
    print("Val feature shape:", X_val_feat.shape)
    print("Positive tiles (train):", y_train.sum(), "/", len(y_train))
    print("Positive tiles (val):  ", y_val.sum(), "/", len(y_val))

    print("\nFitting Logistic Regression baseline...")
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_train_feat, y_train)

    # Evaluate
    val_probs = clf.predict_proba(X_val_feat)[:, 1]
    roc = roc_auc_score(y_val, val_probs)
    pr = average_precision_score(y_val, val_probs)
    print(f"\nBaseline Logistic Regression Results:")
    print(f"ROC-AUC: {roc:.3f}")
    print(f"PR-AUC : {pr:.3f}")

    # Save model
    model_path = os.path.join(models_dir, "baseline_logreg.pkl")
    joblib.dump(clf, model_path)
    print(f"Saved baseline model to {model_path}")

    # Plot ROC and PR curves
    fig_roc, ax_roc = plt.subplots()
    RocCurveDisplay.from_predictions(y_val, val_probs, ax=ax_roc)
    ax_roc.set_title("Logistic Regression ROC (Tile-level)")
    roc_path = os.path.join(curves_dir, "baseline_logreg_roc.png")
    fig_roc.savefig(roc_path, dpi=150, bbox_inches="tight")

    fig_pr, ax_pr = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_val, val_probs, ax=ax_pr)
    ax_pr.set_title("Logistic Regression PR (Tile-level)")
    pr_path = os.path.join(curves_dir, "baseline_logreg_pr.png")
    fig_pr.savefig(pr_path, dpi=150, bbox_inches="tight")

    print(f"ROC curve saved to: {roc_path}")
    print(f"PR curve saved to : {pr_path}")


if __name__ == "__main__":
    main()
