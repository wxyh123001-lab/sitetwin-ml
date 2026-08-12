"""
Offline training script. Run manually, not a long-running service.

Usage:
  python -m ml.train --data training_data/normal_snapshots.pkl

Training data requirement: only a list of Snapshots that are confidently
"normal" is needed, no anomaly labels required (all three models are unsupervised).
"""
import argparse
import pickle
import os
import sys

import joblib
import numpy as np
# IsolationForest / OneClassSVM commented out below -- see the note at their
# training blocks for why. Imports left commented out alongside them so it's
# obvious both halves were disabled together, not just an unused-import slip.
# from sklearn.ensemble import IsolationForest
# from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.features import make_features  # noqa: E402


def train_all(snapshots, config, models_dir):
    os.makedirs(models_dir, exist_ok=True)

    X = np.array([make_features(s) for s in snapshots])
    print(f"Training samples: {X.shape[0]}, feature dimensions: {X.shape[1]}")

    # all three models are sensitive to feature scale (OC-SVM especially), standardize uniformly
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(models_dir, "scaler.joblib"))

    l3cfg = config["l3"]

    # Isolation Forest and OC-SVM disabled: the model-selection experiments
    # (ml/train_iforest.py / ml/train_ocsvm.py / ml/train_lof.py, recorded in
    # SiteTwin_L3模型超参数搜索实验记录_第一轮25trial.docx) showed LOF detecting
    # substantially more injected anomalies than Isolation Forest at the same
    # threshold across repeated trials, so LOF was chosen as the sole L3 model.
    # Left commented out (not deleted) in case that decision gets revisited.
    #
    # iforest = IsolationForest(
    #     n_estimators=l3cfg["n_estimators"],
    #     contamination=l3cfg["contamination"],
    #     random_state=42,
    # )
    # iforest.fit(X_scaled)
    # joblib.dump(iforest, os.path.join(models_dir, "isolation_forest.joblib"))
    # print("Isolation Forest training complete")
    #
    # ocsvm = OneClassSVM(nu=l3cfg["contamination"], kernel="rbf", gamma="scale")
    # ocsvm.fit(X_scaled)
    # joblib.dump(ocsvm, os.path.join(models_dir, "ocsvm.joblib"))
    # print("One-Class SVM training complete")

    # LOF needs novelty=True to score new data after training (otherwise it can only score the training set itself)
    # n_neighbors=46: confirmed via Optuna search (see ml/train_lof.py and
    # SiteTwin_L3模型超参数搜索实验记录_第一轮25trial.docx) -- converged after 200 trials,
    # was previously a placeholder default of 20
    lof = LocalOutlierFactor(n_neighbors=46, novelty=True,
                              contamination=l3cfg["contamination"])
    lof.fit(X_scaled)
    joblib.dump(lof, os.path.join(models_dir, "lof.joblib"))
    print("LOF training complete")

    # Score calibration: instead of a hand-tuned sigmoid scaling coefficient
    # (prone to distortion), save the training set's own raw score distribution
    # and use "percentile rank" at inference time to convert it into a 0~1
    # anomaly score -- more robust and easier to explain:
    # "this point's raw score is more extreme than X percent of the normal points in the training set"
    # iforest_raw = iforest.score_samples(X_scaled)
    # ocsvm_raw = ocsvm.decision_function(X_scaled)
    lof_raw = lof.decision_function(X_scaled)

    # np.save(os.path.join(models_dir, "iforest_raw_dist.npy"), np.sort(iforest_raw))
    # np.save(os.path.join(models_dir, "ocsvm_raw_dist.npy"), np.sort(ocsvm_raw))
    np.save(os.path.join(models_dir, "lof_raw_dist.npy"), np.sort(lof_raw))
    print("Score calibration distributions saved")

    return {"lof": lof, "scaler": scaler}



if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="pickle file containing a list of Snapshots")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(args.data, "rb") as f:
        snapshots = pickle.load(f)

    train_all(snapshots, config, config["l3"]["models_dir"])
