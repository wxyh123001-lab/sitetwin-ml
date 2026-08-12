"""
Aggregator: runs all three per-model hyperparameter searches
(train_iforest / train_ocsvm / train_lof) against the same training data and
shows one summary table.

Scope reminder: this still only trains -- no L0-L3 inference, no
injected-anomaly evaluation. The "best_objective" column is each model's own
internal cross-validation gap (see experiment_utils.py), not a detection
rate. Comparing models on that number tells you which one generalizes best
to held-out normal data with its own hyperparameters, not which one is best
at catching real anomalies -- that comparison needs a separate,
inference-based experiment (deliberately out of scope here).

Usage: python -m ml.train_base [--trials 25]
   or: python -m ml.train_base --iforest-trials 15 --ocsvm-trials 15 --lof-trials 35
       (per-model override; falls back to --trials for any model not given explicitly)
"""
import argparse
import os

import yaml

from ml.experiment_utils import load_training_matrix, render_results_table
from ml import train_iforest, train_ocsvm, train_lof

EXPERIMENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments")

MODEL_MODULES = [train_iforest, train_ocsvm, train_lof]


def run_all(data_path, config, n_trials=25, trials_by_model=None,
            show_plots_per_model=False, show_table=True):
    """
    n_trials: default trial count used for any model not listed in trials_by_model.
    trials_by_model: optional {model_name: n_trials} to override individual models,
        e.g. {"isolation_forest": 15, "ocsvm": 15, "lof": 35}.
    """
    trials_by_model = trials_by_model or {}
    X_scaled, scaler = load_training_matrix(data_path)
    contamination = config["l3"]["contamination"]
    save_dir = os.path.join(EXPERIMENT_DIR, "models")
    plots_dir = os.path.join(EXPERIMENT_DIR, "plots")

    results = []
    for mod in MODEL_MODULES:
        model_trials = trials_by_model.get(mod.MODEL_NAME, n_trials)
        print(f"--- running {mod.MODEL_NAME} search ({model_trials} trials) ---")
        r = mod.run(X_scaled, contamination, n_trials=model_trials,
                    show_plots=show_plots_per_model, save_dir=save_dir, plots_dir=plots_dir)
        results.append(r)
        print(f"  best_params={r['best_params']}, best_objective={r['best_objective']:.4f}, "
              f"fit_time={r['fit_time']:.3f}s")

    headers = ["model", "trials", "best_params", "cv objective\n(|gap| vs contamination)",
               "train score\nmean", "train score\nstd", "fit time (s)"]
    rows = [
        (r["model_name"], str(trials_by_model.get(r["model_name"], n_trials)),
         _format_params(r["best_params"]), f"{r['best_objective']:.4f}",
         f"{r['train_score_mean']:.4f}", f"{r['train_score_std']:.4f}", f"{r['fit_time']:.3f}")
        for r in results
    ]
    os.makedirs(plots_dir, exist_ok=True)
    render_results_table(
        rows, headers,
        f"L3 model comparison -- training-only, contamination={contamination}\n"
        f"NOTE: this ranks generalization on held-out NORMAL data, "
        f"not detection rate on real anomalies",
        show=show_table,
        save_path=os.path.join(plots_dir, "comparison_table.png"),
    )
    return results


def _format_params(params):
    return "\n".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in params.items())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="training_data/normal_snapshots.pkl")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--trials", type=int, default=25,
                         help="default trial count for any model not overridden below")
    parser.add_argument("--iforest-trials", type=int, default=None)
    parser.add_argument("--ocsvm-trials", type=int, default=None)
    parser.add_argument("--lof-trials", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    trials_by_model = {}
    if args.iforest_trials is not None:
        trials_by_model["isolation_forest"] = args.iforest_trials
    if args.ocsvm_trials is not None:
        trials_by_model["ocsvm"] = args.ocsvm_trials
    if args.lof_trials is not None:
        trials_by_model["lof"] = args.lof_trials

    run_all(args.data, config, n_trials=args.trials, trials_by_model=trials_by_model)
