"""
Local Outlier Factor (novelty=True) hyperparameter search + final training.
Standalone: `python -m ml.train_lof` trains directly from
training_data/normal_snapshots.pkl.

Search space: n_neighbors. contamination is fixed to the configured value,
same reasoning as the other two scripts -- shared assumption, not a
per-model tuning knob. novelty=True is fixed (required to score new data
after training; without it LOF can only score the training set itself).
"""
import os
import time

import joblib
import numpy as np
import optuna
from sklearn.neighbors import LocalOutlierFactor

from ml.experiment_utils import (
    load_training_matrix, cv_contamination_gap,
    plot_score_histogram, plot_optuna_convergence,
)

MODEL_NAME = "lof"
EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "ml", "experiments")


def _build(trial, contamination):
    n_neighbors = trial.suggest_int("n_neighbors", 5, 100)
    return LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination=contamination)


def run(X_scaled, contamination, n_trials=25, n_splits=5, show_plots=True,
        save_dir=None, plots_dir=None, random_state=42):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        return cv_contamination_gap(lambda: _build(trial, contamination),
                                     X_scaled, contamination, n_splits, random_state)

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    t0 = time.perf_counter()
    final_model = LocalOutlierFactor(n_neighbors=best_params["n_neighbors"],
                                      novelty=True, contamination=contamination)
    final_model.fit(X_scaled)
    fit_time = time.perf_counter() - t0

    raw_scores = final_model.decision_function(X_scaled)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(final_model, os.path.join(save_dir, f"{MODEL_NAME}.joblib"))
        np.save(os.path.join(save_dir, f"{MODEL_NAME}_raw_dist.npy"), np.sort(raw_scores))

    if show_plots or plots_dir:
        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
        save_path_hist = os.path.join(plots_dir, f"{MODEL_NAME}_score_hist.png") if plots_dir else None
        save_path_conv = os.path.join(plots_dir, f"{MODEL_NAME}_convergence.png") if plots_dir else None
        plot_score_histogram(raw_scores, f"{MODEL_NAME}: training-set score distribution",
                              show=show_plots, save_path=save_path_hist)
        plot_optuna_convergence(study, f"{MODEL_NAME}: Optuna search convergence "
                                        f"({n_trials} trials)",
                                 show=show_plots, save_path=save_path_conv)

    return {
        "model_name": MODEL_NAME,
        "best_params": best_params,
        "best_objective": study.best_value,
        "train_score_mean": float(np.mean(raw_scores)),
        "train_score_std": float(np.std(raw_scores)),
        "fit_time": fit_time,
        "model": final_model,
        "raw_scores": raw_scores,
    }


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_data/normal_snapshots.pkl")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    X_scaled, scaler = load_training_matrix(args.data)
    result = run(X_scaled, config["l3"]["contamination"], n_trials=args.trials,
                 save_dir=os.path.join(EXPERIMENT_DIR, "models"),
                 plots_dir=os.path.join(EXPERIMENT_DIR, "plots"))

    print(f"best_params: {result['best_params']}")
    print(f"best_objective (|holdout outlier frac - contamination|): {result['best_objective']:.4f}")
    print(f"train_score_mean={result['train_score_mean']:.4f}, "
          f"train_score_std={result['train_score_std']:.4f}, "
          f"fit_time={result['fit_time']:.3f}s")
