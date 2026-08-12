"""
Cold-start (per-site local training) logic.

A model trained at one site does not transfer to another -- temperature / CO2 /
current baselines differ by location -- so the system must train on each site's
own data. On first run there is no model yet: it collects this site's
"confirmed normal" data (running L0-L2 only, no model), and once it has enough,
trains L3 locally and activates it. Subsequent runs just load that model.

"Enough" is judged by the SPAN of the collected data (earliest to latest
timestamp), not wall-clock time -- robust to restarts and to idle periods.
Before trusting the trained model, run_diagnostics() checks that every feature
actually varied during collection, so we never train a model that would flag
never-before-seen-but-normal behavior (e.g. equipment finally running) as
anomalous.
"""
import os
import numpy as np

from ml.features import make_features, FEATURE_NAMES


def models_ready(config):
    """True if a locally-trained LOF model + its calibration artifacts exist."""
    d = config["l3"]["models_dir"]
    return all(os.path.exists(os.path.join(d, f))
               for f in ("lof.joblib", "scaler.joblib", "lof_raw_dist.npy"))


def collected_span_days(snapshots):
    """Span in days between the earliest and latest collected snapshot, from the
    data's own timestamps. Robust to restarts / idle periods. 0.0 if <2."""
    if len(snapshots) < 2:
        return 0.0
    ts = [s.timestamp for s in snapshots]
    return (max(ts) - min(ts)).total_seconds() / 86400.0


def collect_normal(snapshots, config):
    """Run L0-L2 only (no model needed) and return the snapshots that trigger no
    alert -- the 'confirmed normal' subset used for training."""
    from pipeline import Pipeline
    pipeline = Pipeline(config, layers=["L0", "L1", "L2"])
    normal = []
    for snap in snapshots:
        if pipeline.run(snap)["alert_state"] == "normal":
            normal.append(snap)
    return normal


def run_diagnostics(snapshots):
    """
    Check the collected data actually exercised every feature before training on
    it. A feature that never varied (max == min) means that behavior was never
    observed -- training on it would make the model flag the behavior as
    anomalous the first time it occurs. Returns (ok, messages).
    """
    messages = [f"collected {len(snapshots)} normal samples"]
    if len(snapshots) < 2:
        return False, messages + ["not enough samples to train"]

    X = np.array([make_features(s) for s in snapshots])
    ok = True
    for name, col in zip(FEATURE_NAMES, X.T):
        if np.ptp(col) < 1e-9:  # peak-to-peak 0 -> never varied
            ok = False
            messages.append(
                f"WARNING: feature '{name}' never varied during collection -- the "
                f"model has not seen this behavior and would flag it as anomalous; "
                f"keep collecting until it is observed, or investigate the sensor")
    if ok:
        messages.append("diagnostics passed: every feature varied during collection")
    return ok, messages


def train_local_model(snapshots, config):
    """Persist the collected normal data and train L3 (LOF) on it locally.
    Runs diagnostics first; returns True only if it actually trained."""
    import pickle
    from ml.train import train_all

    ok, messages = run_diagnostics(snapshots)
    print("\n".join(messages))
    if not ok:
        print("diagnostics failed -- NOT training, L3 stays inactive (L0-L2 remain active).")
        return False

    os.makedirs("training_data", exist_ok=True)
    with open("training_data/normal_snapshots.pkl", "wb") as f:
        pickle.dump(snapshots, f)
    train_all(snapshots, config, config["l3"]["models_dir"])
    print("local L3 model trained and activated.")
    return True
