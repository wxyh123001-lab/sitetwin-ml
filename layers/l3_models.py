"""
L3 unsupervised learning layer.
The only layer with actual machine-learning behavior (the fit() training step).
This file does inference only: loads the model trained by ml/train.py and
scores new data.

Model selection: Isolation Forest and OC-SVM were dropped after the
hyperparameter-search experiments (ml/train_iforest.py / ml/train_ocsvm.py /
ml/train_lof.py, recorded in SiteTwin_L3模型超参数搜索实验记录_第一轮25trial.docx)
showed LOF detecting substantially more injected anomalies than Isolation
Forest at the same threshold, repeatably across multiple trials -- so LOF is
now the sole L3 model, not just the primary one among three.

Score calibration method: percentile rank, not a hand-tuned sigmoid scaling.
Meaning: the new data point's raw model score is ranked against the training
set's own score distribution; the more extreme/rare the rank, the closer the
anomaly score gets to 1.
The benefit of this approach is that the score scale is determined by the
training data itself, no scaling coefficient needs to be guessed, and it's
more interpretable: "rarer than X percent of the normal points in the training set".
"""
import os
import numpy as np
import joblib

from ml.features import make_features


def _percentile_anomaly_score(raw_value, sorted_normal_dist):
    """
    A higher raw_value = more normal (true for LOF's decision_function).
    Returns: raw_value's rank (0~1) within the training distribution; the lower
    the rank (further toward the left/less-normal end of the distribution), the
    higher the anomaly score should be, hence 1 - rank.
    """
    if len(sorted_normal_dist) == 0:
        return 0.0
    rank = np.searchsorted(sorted_normal_dist, raw_value) / len(sorted_normal_dist)
    return round(float(1.0 - rank), 4)


class L3Layer:
    def __init__(self, config):
        self.cfg = config["l3"]
        self.models_dir = self.cfg["models_dir"]
        self.alert_threshold = self.cfg["score_alert_threshold"]
        self.loaded = False
        self._try_load()

    def _try_load(self):
        try:
            d = self.models_dir
            self.scaler = joblib.load(os.path.join(d, "scaler.joblib"))
            self.lof = joblib.load(os.path.join(d, "lof.joblib"))
            self.lof_dist = np.load(os.path.join(d, "lof_raw_dist.npy"))
            self.loaded = True
        except FileNotFoundError:
            # model not trained yet, L3 stays inactive without affecting L0-L2
            self.loaded = False

    def process(self, snapshot):
        if not self.loaded:
            return snapshot

        x = np.array([make_features(snapshot)])
        x_scaled = self.scaler.transform(x)

        lof_raw = self.lof.decision_function(x_scaled)[0]
        lof_score = _percentile_anomaly_score(lof_raw, self.lof_dist)

        snapshot.anomaly_scores_by_model = {"lof": lof_score}
        snapshot.anomaly_score = lof_score

        if snapshot.anomaly_score >= self.alert_threshold:
            snapshot.add_alert(
                "L3", "l3_rare_pattern", "info",
                f"Rare data combination detected, anomaly score {snapshot.anomaly_score:.2f} (not corroborated by the rule layers)"
            )
        return snapshot
