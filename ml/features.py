"""
Feature engineering. The training script and online inference must call the
same function, otherwise training and inference features diverge and the
model silently fails.
"""
import math

FEATURE_NAMES = [
    "temperature", "humidity", "co2", "voc_index",
    "pir_triggered", "light_lux",
    "vibration_rms", "equip_temp", "current",
    "hour_sin", "hour_cos",
]


def make_features(snapshot, env_pod="pod_01", activity_pod="pod_02", equip_pod="pod_03"):
    """Convert a Snapshot into a numeric feature vector (concatenated across pods)"""
    env = snapshot.readings.get(env_pod, {})
    act = snapshot.readings.get(activity_pod, {})
    equip = snapshot.readings.get(equip_pod, {})
    hour = snapshot.timestamp.hour + snapshot.timestamp.minute / 60.0

    return [
        env.get("temperature", 0.0) or 0.0,
        env.get("humidity", 0.0) or 0.0,
        env.get("co2", 0.0) or 0.0,
        env.get("voc_index", 0.0) or 0.0,
        1.0 if act.get("pir_triggered") else 0.0,
        act.get("light_lux", 0.0) or 0.0,
        equip.get("vibration_rms", 0.0) or 0.0,
        equip.get("equip_temp", 0.0) or 0.0,
        equip.get("current", 0.0) or 0.0,
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
    ]
