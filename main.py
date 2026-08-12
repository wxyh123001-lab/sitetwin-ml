"""
Main entry point. Two run modes, selected with --source:

  python main.py --source sim          (default) -- simulated data, for local
      dev/validation.

  python main.py --source thingsboard  -- poll real telemetry from ThingsBoard.
      Needs a reachable instance and the TB_USERNAME / TB_PASSWORD environment
      variables set; host and the device->pod mapping come from config.yaml's
      `thingsboard` section.

Cold start (both modes): a model trained at one site does not transfer to
another (temperature / CO2 / current baselines differ), so there is NO
pre-shipped model. On the first run the flow is data -> L0-L2 -> collect this
site's normal data -> (once enough) train L3 -> activate. Later runs load that
local model. See cold_start.py for how "enough" (a data-span threshold) and the
pre-training diagnostics are decided.

The pipeline.run(snapshot) step is identical in both modes once L3 is active --
only the data source differs.
"""
import argparse
import os
import time
import yaml
from datetime import datetime

from pipeline import Pipeline
from simulator.generate import generate_day
from cold_start import (
    models_ready, collected_span_days, collect_normal, train_local_model,
)


def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _print_alert(result):
    print(f"[{result['timestamp']}] {result['alert_state'].upper()} "
          f"| score={result['anomaly_score']} "
          f"| rules={result['triggered_rules']}")


def _cold_start_sim(config):
    """First sim run: generate `collection_days` of clean data, keep the L0-L2
    'confirmed normal' subset, run diagnostics, and train L3 locally. Simulated
    time is compressed, so the full collection span is produced in one go and
    the span threshold always passes on the first run. Returns True if trained."""
    days = config["cold_start"]["collection_days"]
    print(f"No local model yet -- cold start: generating {days} days of clean "
          f"simulated data, running L0-L2, then training L3.")
    snapshots = generate_day(
        start_dt=datetime(2026, 8, 1, 0, 0),
        days=days,
        step_minutes=2,  # must be < each pod's offline_timeout, else L0 misreads every step as offline
    )
    normal = collect_normal(snapshots, config)
    print(f"collected span: {collected_span_days(normal):.1f} days "
          f"(threshold {days} days)")
    return train_local_model(normal, config)


def run_with_simulated_data(config):
    # Cold start: on the first run there is no model -- collect + train first.
    if not models_ready(config):
        if not _cold_start_sim(config):
            return  # diagnostics blocked training; L3 not activated
        print("-" * 60)

    # Model is ready -> run the full L0-L3 pipeline on a fresh day that has
    # anomalies injected, to demonstrate detection.
    pipeline = Pipeline(config)
    snapshots = generate_day(
        start_dt=datetime(2026, 8, 1, 0, 0),
        days=1,
        step_minutes=2,
        inject_anomalies=[
            {"at_minute_offset": 600, "type": "co2_spike", "pod": "pod_01"},
            {"at_minute_offset": 900, "type": "equip_stall"},
        ],
    )
    for snap in snapshots:
        result = pipeline.run(snap)
        if result["alert_state"] != "normal":
            _print_alert(result)


COLLECTION_BUFFER = "training_data/collection_buffer.pkl"


def _tb_setup(config):
    """Log in and resolve each device's UUID + SLOT->field map once. Returns
    (client, poll_interval, devices) where devices maps pod_id -> {device_id, slot_map}."""
    from thingsboard_client import ThingsBoardClient
    from converters import build_slot_field_map

    tbcfg = config["thingsboard"]
    username = os.environ.get("TB_USERNAME")
    password = os.environ.get("TB_PASSWORD")
    if not username or not password:
        raise SystemExit(
            "Set TB_USERNAME and TB_PASSWORD environment variables "
            "(see thingsboard_api_reference.md -- credentials must not be hardcoded).")

    client = ThingsBoardClient(tbcfg["host"])
    client.login(username, password)

    devices = {}  # pod_id -> {"device_id": ..., "slot_map": {...}}
    for device_name, pod_id in tbcfg["device_to_pod"].items():
        device_id = client.get_device_id(device_name)
        slot_map = build_slot_field_map(client.get_attributes(device_id))
        devices[pod_id] = {"device_id": device_id, "slot_map": slot_map}
        print(f"resolved {device_name} -> {pod_id} ({device_id}), slots: {slot_map}")

    return client, tbcfg["poll_interval_seconds"], devices


def _tb_poll_snapshot(client, devices, last_ts_ms, now_ms):
    """Pull each pod's new readings in (last_ts_ms, now_ms], and build one
    'current state' Snapshot from the newest reading per pod. Returns the
    Snapshot, or None if no pod reported anything."""
    from converters import from_thingsboard_timeseries, build_snapshot

    readings_by_pod = {}
    for pod_id, dev in devices.items():
        slot_keys = list(dev["slot_map"].keys())
        if not slot_keys:
            continue
        raw = client.get_timeseries(dev["device_id"], slot_keys, last_ts_ms, now_ms)
        parsed = from_thingsboard_timeseries(raw, dev["slot_map"])
        if parsed:
            readings_by_pod[pod_id] = parsed[-1][1]  # newest in window = current state
    if not readings_by_pod:
        return None
    return build_snapshot(datetime.now(), readings_by_pod)


def _load_buffer():
    import pickle
    if os.path.exists(COLLECTION_BUFFER):
        with open(COLLECTION_BUFFER, "rb") as f:
            return pickle.load(f)
    return []


def _save_buffer(snapshots):
    import pickle
    os.makedirs("training_data", exist_ok=True)
    with open(COLLECTION_BUFFER, "wb") as f:
        pickle.dump(snapshots, f)


def _cold_start_thingsboard(config, client, poll_interval, devices):
    """
    Collection phase for real data: poll, keep the L0-L2 'confirmed normal'
    snapshots in a disk-backed buffer, and once the buffer SPANS
    collection_days (earliest to latest timestamp, so a restart resumes and idle
    time doesn't count), run diagnostics and train. The buffer persists across
    restarts. Returns once L3 is trained + activated.
    """
    days = config["cold_start"]["collection_days"]
    recheck_interval = config["cold_start"]["recheck_interval_hours"] * 3600
    l2_pipeline = Pipeline(config, layers=["L0", "L1", "L2"])  # no model yet
    buffer = _load_buffer()
    print(f"No local model yet -- cold start: collecting normal data until it "
          f"spans {days} days (buffer has {len(buffer)} samples, "
          f"{collected_span_days(buffer):.1f} days so far).")

    last_ts_ms = int(time.time() * 1000)
    last_train_attempt = 0.0  # wall-clock of the last train/diagnose attempt
    while True:
        now_ms = int(time.time() * 1000)
        snap = _tb_poll_snapshot(client, devices, last_ts_ms, now_ms)
        last_ts_ms = now_ms

        if snap is not None and l2_pipeline.run(snap)["alert_state"] == "normal":
            buffer.append(snap)
            _save_buffer(buffer)
            span = collected_span_days(buffer)
            print(f"collected {len(buffer)} normal samples, span {span:.2f}/{days} days")
            # Once the span threshold is met, try to train. If diagnostics fail
            # (a feature never varied), don't retry every poll -- re-check only
            # once per recheck_interval, since the result won't change until the
            # missing behavior actually shows up in newly collected data.
            if span >= days and (time.time() - last_train_attempt) >= recheck_interval:
                last_train_attempt = time.time()
                if train_local_model(buffer, config):
                    return  # L3 trained + activated
                print(f"diagnostics not passed -- re-checking in "
                      f"{recheck_interval / 3600:.0f}h while collecting continues.")

        time.sleep(poll_interval)


def run_with_thingsboard_data(config):
    """
    Poll ThingsBoard on a fixed interval. On first deployment there is no model,
    so it runs the cold-start collection phase first; once L3 is trained it drops
    into the steady-state loop where each poll's newest reading per pod becomes a
    'current state' Snapshot and the trend-based L2 scenarios build history across
    successive polls.

    Not yet tested against a live instance -- structurally complete, needs real
    host/credentials to actually run.
    """
    client, poll_interval, devices = _tb_setup(config)

    if not models_ready(config):
        _cold_start_thingsboard(config, client, poll_interval, devices)
        print("-" * 60)

    pipeline = Pipeline(config)  # full L0-L3
    last_ts_ms = int(time.time() * 1000)  # only pull data from now onward
    print(f"polling ThingsBoard every {poll_interval}s ...")
    while True:
        now_ms = int(time.time() * 1000)
        snap = _tb_poll_snapshot(client, devices, last_ts_ms, now_ms)
        last_ts_ms = now_ms
        if snap is not None:
            result = pipeline.run(snap)
            if result["alert_state"] != "normal":
                _print_alert(result)
        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sim", "thingsboard"], default="sim",
                         help="sim = simulated data (default); "
                              "thingsboard = poll real data from ThingsBoard")
    args = parser.parse_args()

    config = load_config()
    if args.source == "sim":
        run_with_simulated_data(config)
    else:
        run_with_thingsboard_data(config)
