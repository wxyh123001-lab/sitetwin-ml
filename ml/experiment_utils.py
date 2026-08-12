"""
Shared plumbing for the per-model hyperparameter-search scripts
(train_iforest.py / train_ocsvm.py / train_lof.py) and the aggregator
(train_base.py).

Scope reminder (agreed before writing this): these scripts search
hyperparameters and fit final models. They deliberately do NOT run L0-L3
inference or evaluate.py-style detection-rate scoring -- that is a separate,
later concern. Model quality here is judged purely from held-out "confirmed
normal" training data, never from injected-anomaly test data.

Objective function (agreed direction: Plan A): for a given hyperparameter
choice, k-fold cross-validate on the training data. In each fold, fit on the
train split, then use the fitted model's own .predict() (which applies its
contamination/nu-derived threshold) on the held-out split. A well-generalizing
model should flag roughly `contamination` fraction of held-out normal points
as outliers -- no more, no less. The objective is the mean absolute gap
between that observed fraction and the configured contamination, averaged
across folds. Optuna minimizes this.
"""
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ml.features import make_features


def load_training_matrix(pkl_path):
    """Load the training Snapshot list and return (X_scaled, scaler)."""
    with open(pkl_path, "rb") as f:
        snapshots = pickle.load(f)
    X = np.array([make_features(s) for s in snapshots])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def cv_contamination_gap(build_fn, X_scaled, contamination, n_splits=5, random_state=42):
    """
    build_fn(): -> a fresh unfitted estimator for one trial's hyperparameters.
    Returns the mean |observed_outlier_fraction - contamination| across folds.
    Lower is better (Optuna direction="minimize").
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gaps = []
    for train_idx, holdout_idx in kf.split(X_scaled):
        model = build_fn()
        model.fit(X_scaled[train_idx])
        pred = model.predict(X_scaled[holdout_idx])  # sklearn convention: -1 outlier, 1 inlier
        observed_fraction = float(np.mean(pred == -1))
        gaps.append(abs(observed_fraction - contamination))
    return float(np.mean(gaps))


def plot_score_histogram(scores, title, show=True, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=40, color="#4C72B0", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("raw model score (higher = more normal)")
    ax.set_ylabel("count")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_optuna_convergence(study, title, show=True, save_path=None):
    values = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.minimum.accumulate(values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(values) + 1), values, "o", alpha=0.4, label="trial value")
    ax.plot(range(1, len(best_so_far) + 1), best_so_far, "-", color="#C44E52", label="best so far")
    ax.set_title(title)
    ax.set_xlabel("trial")
    ax.set_ylabel("objective (lower is better)")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def render_results_table(rows, headers, title, show=True, save_path=None):
    """rows: list of tuples of strings, already formatted for display.
    Cell text may contain '\\n' for multi-line wrapping (used for the
    best_params column, which can otherwise overflow into neighboring cells)."""
    # width per column sized to its longest line (header or any cell), not a
    # fixed guess -- avoids the overflow that a uniform column width produces
    # once one column's content (e.g. best_params) is much longer than the rest
    n_cols = len(headers)
    col_char_width = [
        max(len(line) for cell in ([headers[c]] + [row[c] for row in rows])
            for line in str(cell).split("\n"))
        for c in range(n_cols)
    ]
    total_width = sum(col_char_width)
    col_fractions = [w / total_width for w in col_char_width]

    fig_width = max(10, total_width * 0.11)
    max_lines_per_row = max((max(str(cell).count("\n") for cell in row) + 1 for row in rows), default=1)
    fig_height = 1.5 + 0.5 * len(rows) * max_lines_per_row
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center",
                      colWidths=col_fractions)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6 * max_lines_per_row)
    ax.set_title(title, pad=20)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
