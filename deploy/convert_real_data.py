"""
Converts the 3 pod*_full.json dumps (from deploy/fetch_all_data.sh -- full
historical ThingsBoard timeseries per pod) into a list[Snapshot] pickle, ready
for `python -m ml.train --data <output>`.

Unlike the live poll loop (which only has a short recent window to work
with), we have each pod's ENTIRE history here, so gating is done precisely:
for every raw reading, look at that field's alarm-active history and use
whatever was the most recent transition AT OR BEFORE that exact reading's own
timestamp (not "the current window's last known state" -- we don't need that
approximation when the full timeline is available).

One Snapshot is emitted per distinct timestamp seen across ANY pod's raw
readings, with every field forward-filled to its latest known (and gated)
value as of that timestamp -- mirrors how the live polling loop's fold-merge
works, just applied across the whole history instead of one poll window.

Usage:
    python deploy/convert_real_data.py
    python -m ml.train --data training_data/real_snapshots.pkl
"""
import json
import os
import pickle
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from state import Snapshot  # noqa: E402
from simulator.tb_mock import POD_SCHEMA  # noqa: E402

_BOOL_FIELDS = {"pir_triggered", "door_state"}

POD_FILES = {
    "pod_01": "pod1_full.json",
    "pod_02": "pod2_full.json",
    "pod_03": "pod3_full.json",
}


def _cast(field, raw_value):
    if field in _BOOL_FIELDS:
        try:
            return float(raw_value) != 0.0
        except (TypeError, ValueError):
            return str(raw_value).strip().lower() == "true"
    value = float(raw_value)
    if field == "current":
        value = abs(value)  # ina219_current can be negative, see converters._cast_value
    return value


def _active_at_or_before(active_points, ts_ms):
    """Most recent active-state at or before ts_ms. active_points must be
    sorted ascending by ts. Empty/no evidence yet -> False (fail-open, same
    default as converters._is_currently_active)."""
    state = False
    for p in active_points:
        if p["ts"] > ts_ms:
            break
        state = str(p["value"]).strip().lower() == "true"
    return state


def load_pod_fields(json_path, pod_id):
    """Returns (readings_by_field, gates_by_field):
      readings_by_field: {our_field: sorted [(ts_ms, cast_value), ...]}
      gates_by_field: {our_field: sorted [(ts_ms, raw_active_value), ...]} for
        fields whose capability has a rule -- the raw alarm_..._active points.
    """
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    readings_by_field = {}
    gates_by_field = {}
    for sensor_id, capability, field, rule_kind, _threshold in POD_SCHEMA[pod_id]:
        points = sorted(raw.get(sensor_id, []), key=lambda p: p["ts"])
        readings_by_field[field] = [(p["ts"], _cast(field, p["value"])) for p in points]
        if rule_kind:
            active_key = f"alarm_{capability}_{rule_kind}_active"
            gates_by_field[field] = sorted(raw.get(active_key, []), key=lambda p: p["ts"])

    return readings_by_field, gates_by_field


def build_pod_timeline(readings_by_field, gates_by_field):
    """Returns {ts_ms: {field: value}} -- one entry per distinct raw-reading
    timestamp for THIS pod, forward-filling every field to its latest known
    value as of that timestamp, with gating applied using each reading's own
    timestamp against that field's full alarm history."""
    all_ts = sorted({ts for pts in readings_by_field.values() for ts, _ in pts})
    idx = {f: -1 for f in readings_by_field}
    current_val = {f: None for f in readings_by_field}

    timeline = {}
    for ts in all_ts:
        for field, pts in readings_by_field.items():
            while idx[field] + 1 < len(pts) and pts[idx[field] + 1][0] <= ts:
                idx[field] += 1
                current_val[field] = pts[idx[field]][1]
        fields = {}
        for field, val in current_val.items():
            if val is None:
                continue
            gate_points = gates_by_field.get(field)
            if gate_points and _active_at_or_before(gate_points, ts):
                continue  # this field's capability is currently flagged -- drop it
            fields[field] = val
        timeline[ts] = fields

    return timeline


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    per_pod_timeline = {}
    for pod_id, filename in POD_FILES.items():
        path = os.path.join(here, filename)
        readings_by_field, gates_by_field = load_pod_fields(path, pod_id)
        per_pod_timeline[pod_id] = build_pod_timeline(readings_by_field, gates_by_field)
        print(f"{pod_id}: {len(per_pod_timeline[pod_id])} distinct reading timestamps")

    # merge across pods: one Snapshot per distinct timestamp seen on ANY pod,
    # each pod forward-filled to its own latest known (gated) reading as of
    # that global timestamp.
    all_ts = sorted({ts for tl in per_pod_timeline.values() for ts in tl})
    pod_ts_lists = {pod_id: sorted(tl.keys()) for pod_id, tl in per_pod_timeline.items()}
    pod_idx = {pod_id: -1 for pod_id in per_pod_timeline}
    pod_current = {pod_id: {} for pod_id in per_pod_timeline}

    snapshots = []
    for ts in all_ts:
        for pod_id, ts_list in pod_ts_lists.items():
            while pod_idx[pod_id] + 1 < len(ts_list) and ts_list[pod_idx[pod_id] + 1] <= ts:
                pod_idx[pod_id] += 1
                pod_current[pod_id] = per_pod_timeline[pod_id][ts_list[pod_idx[pod_id]]]
        readings = {pod_id: dict(fields) for pod_id, fields in pod_current.items() if fields}
        if readings:
            snapshots.append(Snapshot(timestamp=datetime.fromtimestamp(ts / 1000.0), readings=readings))

    print(f"total merged snapshots: {len(snapshots)}")
    if snapshots:
        span_days = (snapshots[-1].timestamp - snapshots[0].timestamp).total_seconds() / 86400.0
        print(f"time span: {span_days:.2f} days ({snapshots[0].timestamp} -> {snapshots[-1].timestamp})")

    os.makedirs("training_data", exist_ok=True)
    out_path = "training_data/real_snapshots.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(snapshots, f)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
