"""
Simulated data generator. Used to produce a physically plausible Snapshot
sequence and exercise the whole pipeline before real hardware is available.

Not just random numbers made up: temperature follows a day/night curve, CO2
is generated with a simplified accumulation/decay model driven by "is someone
present", PIR triggers with a probability tied to working hours -- these
parameters should later be replaced with real literature values or measured
calibration values.
"""
import math
import random
from datetime import datetime, timedelta

from state import Snapshot


def _temp_curve(hour, base=22.0, amplitude=2.0):
    return base + amplitude * math.sin((hour - 6) / 24 * 2 * math.pi)


def generate_day(start_dt, days=1, step_minutes=5, seed=42, inject_anomalies=None):
    """
    Generate a multi-day simulated data stream.
    inject_anomalies: list of dict, e.g.
      [{"at_minute_offset": 500, "type": "co2_spike", "pod": "pod_01"}]
    Add "duration_minutes" to hold the anomalous reading across a range of
    consecutive samples instead of a single point -- needed for the trend-based
    L2 scenarios, which compare against history already sitting in their window
    rather than any single sample. A constant elevated reading is enough to
    register as "rising" the moment it enters the window; no gradual ramp needed.
    """
    rng = random.Random(seed)
    snapshots = []

    co2_level = 500.0
    occupied_ground_truth = False
    door_is_open = False

    total_steps = int(days * 24 * 60 / step_minutes)
    for i in range(total_steps):
        ts = start_dt + timedelta(minutes=i * step_minutes)
        hour = ts.hour + ts.minute / 60.0
        # daily active hours (8am-7pm), every day of the week -- the site/equipment
        # runs 7 days a week, not just Mon-Fri, so there's deliberately no
        # weekday check here (there used to be one; removed)
        is_work_hour = 8 <= hour <= 19

        # occupancy ground truth (used only to generate data, not a system input)
        if is_work_hour:
            occupied_ground_truth = rng.random() < 0.75
        else:
            occupied_ground_truth = rng.random() < 0.03

        # CO2: rises slowly when occupied, decays back to baseline when unoccupied (simplified first-order model)
        target = 900.0 if occupied_ground_truth else 480.0
        co2_level += (target - co2_level) * 0.08 + rng.gauss(0, 5)
        co2_level = max(420.0, co2_level)

        temperature = _temp_curve(hour) + (1.0 if occupied_ground_truth else 0.0) + rng.gauss(0, 0.3)
        humidity = 45 + rng.gauss(0, 3)
        voc = 40 + (30 if occupied_ground_truth else 0) + rng.gauss(0, 5)
        voc = max(10, voc)

        pir = occupied_ground_truth and rng.random() < 0.5
        # door_state is the reed switch's native continuous open/closed reading (not an
        # edge-triggered event): once open it tends to stay open for a few steps before
        # probabilistically closing again, unlike a sparse independent per-step flag
        if door_is_open:
            if rng.random() < 0.15:
                door_is_open = False
        else:
            if rng.random() < (0.03 if occupied_ground_truth else 0.005):
                door_is_open = True
        light_lux = (300 if (occupied_ground_truth or is_work_hour) else 5) + rng.gauss(0, 10)

        equip_running = is_work_hour and rng.random() < 0.6
        # current: confirmed unit is mA. ~200mA running / near-0 idle is a placeholder
        # small-motor magnitude, not a calibrated value -- still needs real hardware data.
        current = (200.0 + rng.gauss(0, 20)) if equip_running else max(0, rng.gauss(0, 2))
        # vibration_rms: unit is g, not m/s^2 as previously assumed -- corrected per
        # TB_Data_Reference_for_ML.md (real capability name is vibration_rms_g).
        # Delivered pre-computed by the sensor/firmware from the X/Y/Z axes (no
        # aggregation logic needed on our end); assumed gravity-removed (standard
        # practice for vibration RMS, distinct from raw acceleration magnitude) --
        # baseline near 0 at rest, not near 1. These simulated magnitudes were
        # tuned by feel, not against real g-scale data -- may need revisiting.
        vibration = (0.15 + rng.gauss(0, 0.02)) if equip_running else max(0, rng.gauss(0, 0.005))
        equip_temp = _temp_curve(hour) + (5 if equip_running else 0) + rng.gauss(0, 0.5)

        readings = {
            "pod_01": {
                "temperature": round(temperature, 2),
                "humidity": round(max(0, humidity), 2),
                "co2": round(co2_level, 1),
                "voc_index": round(voc, 1),
                "battery_pct": 90,
            },
            "pod_02": {
                "pir_triggered": pir,
                "door_state": door_is_open,
                "light_lux": round(max(0, light_lux), 1),
                "battery_pct": 88,
            },
            "pod_03": {
                "vibration_rms": round(vibration, 3),
                "equip_temp": round(equip_temp, 2),
                "current": round(current, 3),
                "battery_pct": 95,
            },
        }

        if inject_anomalies:
            for anomaly in inject_anomalies:
                start = anomaly["at_minute_offset"]
                end = start + anomaly.get("duration_minutes", 0)
                if start <= i * step_minutes <= end:
                    _apply_anomaly(readings, anomaly)

        snapshots.append(Snapshot(timestamp=ts, readings=readings))

    return snapshots


def _apply_anomaly(readings, anomaly):
    """
    Artificially injected anomaly, applied to every sample within
    [at_minute_offset, at_minute_offset + duration_minutes]. Point-in-time types
    (co2_spike/temp_spike/equip_stall/voc_event/sensor_fault) set an absolute value
    and typically use duration_minutes=0 (default), matching L1/L0's instantaneous
    checks. Trend/duration types add a delta on top of whatever the baseline
    already is, so the injected reading stays elevated relative to history
    regardless of time-of-day baseline drift, and need duration_minutes to cover
    at least the target scenario's window_minutes/duration_minutes in config.yaml.
    """
    t = anomaly["type"]
    pod = anomaly.get("pod", "pod_01")
    if t == "co2_spike":
        readings[pod]["co2"] = 2600
    elif t == "temp_spike":
        readings[pod]["temperature"] = 33
    elif t == "equip_stall":
        readings["pod_03"]["current"] = 600  # mA, above the "high" current_label band
        readings["pod_03"]["vibration_rms"] = 0.0
    elif t == "voc_event":
        readings["pod_01"]["voc_index"] = 300
    elif t == "sensor_fault":
        readings[pod]["temperature"] = 200  # physically impossible value, triggers L0
    elif t == "door_stuck":
        readings["pod_02"]["door_state"] = True
    elif t == "vibration_fault":
        readings["pod_03"]["vibration_rms"] = 1.5
        readings["pod_03"]["current"] = 200  # mA, stays in the "normal" band
    elif t == "equip_overheat":
        # equip_temp + current rising together with elevated VOC -> equip_overheat_air_quality
        readings["pod_03"]["equip_temp"] += 10
        readings["pod_03"]["current"] = max(readings["pod_03"]["current"], 200) * 1.8 + 5
        readings["pod_01"]["voc_index"] = 300
    elif t == "cooling_failure":
        # equip_temp rising, current left untouched -> equip_cooling_failure
        readings["pod_03"]["equip_temp"] += 10
    elif t == "violent_event":
        # temperature + equip_temp + vibration all jump together -> rapid_multi_signal_spike
        readings["pod_01"]["temperature"] += 5
        readings["pod_03"]["equip_temp"] += 8
        readings["pod_03"]["vibration_rms"] = 1.0
