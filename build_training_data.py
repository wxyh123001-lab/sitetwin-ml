"""
Generate L3 training data: simulate multiple days of "normal" data -> filter -> save.
Implements the "one-shot gatekeeping" approach discussed earlier:
  run L0+L1+L2 over historical data, keep only the records where none of the
  three layers fired, as the training set.
"""
import pickle
from datetime import datetime

import yaml

from pipeline import Pipeline
from simulator.generate import generate_day


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # L3 isn't needed during training (no model yet), only L0-L2 filtering
    pipeline = Pipeline(config, layers=["L0", "L1", "L2"])

    snapshots = generate_day(
        start_dt=datetime(2026, 7, 1, 0, 0),
        days=14,          # two weeks of simulated data
        step_minutes=2,
        inject_anomalies=None,  # no anomalies injected into training data
        seed=7,
    )

    normal_snapshots = []
    for snap in snapshots:
        result = pipeline.run(snap)
        if result["alert_state"] == "normal":
            normal_snapshots.append(snap)

    print(f"Total samples: {len(snapshots)}, normal samples after filtering: {len(normal_snapshots)}")

    with open("training_data/normal_snapshots.pkl", "wb") as f:
        pickle.dump(normal_snapshots, f)
    print("Saved to training_data/normal_snapshots.pkl")


if __name__ == "__main__":
    main()
