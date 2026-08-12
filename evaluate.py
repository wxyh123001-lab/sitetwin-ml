"""
Ablation experiment: compares detection rate across three pipeline configs
L1-only / L1+L2 / all layers (with L3) on the same batch of "artificially
injected" test data.

This is the core output for the evaluation section - the point isn't to
compare how fancy the models are, it's to quantify how much detection
capability each layer adds.
"""
from datetime import datetime
import yaml

from pipeline import Pipeline
from simulator.generate import generate_day


def build_test_set(config, seed=99):
    """
    Generate one day of data, injecting each anomaly type once, and record each
    injection's time window (not a single point) to compute detection rate.
    Point-in-time types (matching L0/L1's instantaneous checks) get
    duration_minutes=0; trend/duration types get a duration read straight from
    config.yaml's own scenario window so the injection lasts long enough for
    the target scenario's history buffer to build up -- a constant elevated
    reading registers as "rising" the moment it enters the window, no gradual
    ramp needed, but the window has to actually contain it.
    """
    scen = config["scenarios"]
    injections = [
        {"at_minute_offset": 120, "type": "co2_spike", "pod": "pod_01"},
        {"at_minute_offset": 240, "type": "temp_spike", "pod": "pod_01"},
        {"at_minute_offset": 360, "type": "equip_stall"},
        {"at_minute_offset": 480, "type": "voc_event"},
        {"at_minute_offset": 600, "type": "sensor_fault", "pod": "pod_01"},
        {"at_minute_offset": 720, "type": "vibration_fault"},
        {"at_minute_offset": 840, "type": "door_stuck",
         "duration_minutes": scen["door_left_open"]["duration_minutes"] + 4},
        {"at_minute_offset": 960, "type": "equip_overheat",
         "duration_minutes": scen["equip_overheat_air_quality"]["window_minutes"] + 4},
        {"at_minute_offset": 1080, "type": "cooling_failure",
         "duration_minutes": scen["equip_cooling_failure"]["window_minutes"] + 4},
        {"at_minute_offset": 1200, "type": "violent_event",
         "duration_minutes": scen["rapid_multi_signal_spike"]["window_minutes"] + 4},
    ]
    snapshots = generate_day(
        start_dt=datetime(2026, 9, 1, 0, 0),
        days=1, step_minutes=2, seed=seed, inject_anomalies=injections,
    )
    base_ts = datetime(2026, 9, 1, 0, 0).timestamp()
    injection_windows = [
        (base_ts + inj["at_minute_offset"] * 60,
         base_ts + (inj["at_minute_offset"] + inj.get("duration_minutes", 0)) * 60)
        for inj in injections
    ]
    return snapshots, injection_windows


def run_config(config, layers, snapshots, injection_windows):
    pipeline = Pipeline(config, layers=layers)
    detected = [False] * len(injection_windows)
    fp = 0
    for snap in snapshots:
        result = pipeline.run(snap)
        ts = snap.timestamp.timestamp()
        fired = result["alert_state"] != "normal"
        in_any_window = False
        for idx, (start, end) in enumerate(injection_windows):
            if start <= ts <= end:
                in_any_window = True
                if fired:
                    detected[idx] = True
        if fired and not in_any_window:
            fp += 1

    tp = sum(detected)
    total_injected = len(injection_windows)
    detection_rate = tp / total_injected if total_injected else 0
    return {
        "layers": "+".join(layers),
        "detected": tp, "missed": total_injected - tp, "total_injected": total_injected,
        "detection_rate": round(detection_rate, 3),
        "false_positives": fp,
    }


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    snapshots, injection_windows = build_test_set(config)

    configs = [
        ["L0", "L1"],
        ["L0", "L1", "L2"],
        ["L0", "L1", "L2", "L3"],
    ]

    print(f"{'Pipeline config':<20}{'Detected/Total':<15}{'Detection rate':<10}{'False positives':<10}")
    for layers in configs:
        r = run_config(config, layers, snapshots, injection_windows)
        print(f"{r['layers']:<20}{r['detected']}/{r['total_injected']:<13}"
              f"{r['detection_rate']:<10}{r['false_positives']:<10}")


if __name__ == "__main__":
    main()
