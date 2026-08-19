"""
Main entry point. Three run modes, selected with --source:

  python main.py --source sim              (default) -- simulated data, built
      directly as Snapshots (simulator/generate.py), for local dev/validation.

  python main.py --source sim-thingsboard  -- same simulated ground truth, but
      round-tripped through a fake ThingsBoard REST response (simulator/tb_mock.py,
      schema confirmed against real hardware captures) and parsed by the REAL
      converters.py code (build_sensor_field_map / from_thingsboard_timeseries)
      instead of built directly. Exercises the actual parsing/gating logic
      (including the hardware-alarm-active filtering) without needing a live
      instance. Runs as a fast batch, no real-time sleep, no credentials needed.

  python main.py --source thingsboard      -- poll real telemetry from a live
      ThingsBoard instance. Needs a reachable instance and the TB_USERNAME /
      TB_PASSWORD environment variables set; host and the device->pod mapping
      come from config.yaml's `thingsboard` section.

L0 (data gatekeeper) and L1 (hard limits) are NOT run here -- the hardware side
already handles data-quality gatekeeping and hard-limit alerting, so this
pipeline only runs L2 (context/scenario rules) and L3 (ML anomaly detection).

Cold start (both modes): a model trained at one site does not transfer to
another (temperature / CO2 / current baselines differ), so there is NO
pre-shipped model. On the first run the flow is data -> L2 -> collect this
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
    """First sim run: generate `collection_days` of clean data, keep the L2
    'confirmed normal' subset (L0/L1 are not run -- hardware already handles
    that), run diagnostics, and train L3 locally. Simulated time is compressed,
    so the full collection span is produced in one go and the span threshold
    always passes on the first run. Returns True if trained."""
    days = config["cold_start"]["collection_days"]
    print(f"No local model yet -- cold start: generating {days} days of clean "
          f"simulated data, running L2, then training L3.")
    snapshots = generate_day(
        start_dt=datetime(2026, 8, 1, 0, 0),
        days=days,
        step_minutes=2,
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

    # Model is ready -> run L2-L3 on a fresh day that has anomalies injected, to
    # demonstrate detection. L0/L1 are not run here: the hardware side already
    # handles data-quality gatekeeping and hard-limit alerting, so this pipeline
    # only does context/scenario rules (L2) and ML anomaly detection (L3).
    pipeline = Pipeline(config, layers=["L2", "L3"])
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


def _mock_tb_devices(client):
    """Resolve field_map/gate_map for all 3 pods against the mock client's
    fake attributes -- same shape _tb_setup builds for the real client."""
    from converters import build_sensor_field_map

    devices = {}
    for pod_id in ("pod_01", "pod_02", "pod_03"):
        field_map, gate_map = build_sensor_field_map(client.get_attributes(pod_id), pod_id=pod_id)
        devices[pod_id] = {"device_id": pod_id, "field_map": field_map, "gate_map": gate_map}
    return devices


def _mock_tb_poll_all(client, devices, ground_truth):
    """Batch-drive _tb_poll_snapshot across every distinct timestamp in
    ground_truth (fast, no real-time sleep) -- returns the resulting Snapshots
    in chronological order, each built via the REAL converters.py parsing and
    hardware-alarm gating, not copied directly from ground_truth."""
    ts_list = sorted({int(s.timestamp.timestamp() * 1000) for s in ground_truth})
    parsed = []
    last_ts_ms = ts_list[0] - 1 if ts_list else 0
    for now_ms in ts_list:
        snap = _tb_poll_snapshot(client, devices, last_ts_ms, now_ms)
        last_ts_ms = now_ms
        if snap is not None:
            parsed.append(snap)
    return parsed


def _cold_start_sim_thingsboard(config):
    """Cold start for --source sim-thingsboard: same idea as _cold_start_sim,
    but the generated data is round-tripped through the mock ThingsBoard JSON
    format and the REAL converters.py parsing/gating (including hardware-alarm
    filtering) instead of being used directly as Snapshots."""
    from simulator.tb_mock import MockThingsBoardClient

    days = config["cold_start"]["collection_days"]
    print(f"No local model yet -- cold start (ThingsBoard JSON format): generating "
          f"{days} days of clean simulated data, parsing it through the real "
          f"ThingsBoard code path, running L2, then training L3.")
    ground_truth = generate_day(start_dt=datetime(2026, 8, 1, 0, 0), days=days, step_minutes=2)
    client = MockThingsBoardClient(ground_truth)
    devices = _mock_tb_devices(client)
    parsed = _mock_tb_poll_all(client, devices, ground_truth)
    normal = collect_normal(parsed, config)
    print(f"collected span: {collected_span_days(normal):.1f} days (threshold {days} days)")
    return train_local_model(normal, config)


def run_with_simulated_thingsboard_data(config):
    """
    --source sim-thingsboard: exercises the REAL ThingsBoard parsing/gating
    code path (converters.build_sensor_field_map / from_thingsboard_timeseries,
    _tb_poll_snapshot) against synthetic data shaped exactly like real
    ThingsBoard responses (simulator/tb_mock.py, schema confirmed against
    pod1.json/pod2.json/pod3.json), instead of --source sim's direct-Snapshot
    path. No live instance or credentials needed. Ground truth is the same
    generate_day() --source sim uses, so both modes agree on "what happened" --
    only the wire format and the parsing code differ. Runs as a fast batch, no
    real-time sleep.
    """
    from simulator.tb_mock import MockThingsBoardClient

    if not models_ready(config):
        if not _cold_start_sim_thingsboard(config):
            return
        print("-" * 60)

    # No injected anomalies here on purpose: this mode is about validating
    # normal-data collection + hardware-alarm gating through the real parsing
    # code, not anomaly detection -- that's evaluate.py's job (injects 10
    # anomaly types, including "slightly elevated but not extreme" cases).
    # Whatever alerts appear below are whatever this batch of normal data
    # naturally produced (e.g. a field's alarm happening to be active while
    # everything else is normal is expected, not engineered).
    pipeline = Pipeline(config, layers=["L2", "L3"])
    ground_truth = generate_day(start_dt=datetime(2026, 8, 1, 0, 0), days=1, step_minutes=2)
    client = MockThingsBoardClient(ground_truth)
    devices = _mock_tb_devices(client)
    alerts = 0
    for snap in _mock_tb_poll_all(client, devices, ground_truth):
        result = pipeline.run(snap)
        if result["alert_state"] != "normal":
            alerts += 1
            _print_alert(result)
    if alerts == 0:
        print("(no alerts -- a full day of normal data produced none, as expected)")


COLLECTION_BUFFER = "training_data/collection_buffer.pkl"


def _tb_setup(config):
    """Log in and resolve each device's UUID + reading-key->field map once.
    Returns (client, poll_interval, devices, ml_advisor_device_id) where
    devices maps pod_id -> {device_id, field_map, gate_map}, and
    ml_advisor_device_id is the fixed UUID of the standalone ML_ADVISOR
    device (created once by hand in ThingsBoard) that all ML-originated
    alarms are attached to -- never to a real Pod's own device."""
    from thingsboard_client import ThingsBoardClient
    from converters import build_sensor_field_map

    tbcfg = config["thingsboard"]
    username = os.environ.get("TB_USERNAME")
    password = os.environ.get("TB_PASSWORD")
    if not username or not password:
        raise SystemExit(
            "Set TB_USERNAME and TB_PASSWORD environment variables "
            "(see thingsboard_api_reference.md -- credentials must not be hardcoded).")

    ml_advisor_device_id = tbcfg.get("ml_advisor_device_id")
    if not ml_advisor_device_id:
        raise SystemExit(
            "config.yaml: thingsboard.ml_advisor_device_id is not set. "
            "Create a device named ML_ADVISOR in ThingsBoard first, then put its "
            "UUID here -- ML alarms must never be attached to a real Pod's own device.")

    client = ThingsBoardClient(tbcfg["host"])
    client.login(username, password)

    devices = {}  # pod_id -> {"device_id": ..., "field_map": {...}, "gate_map": {...}}
    for device_name, pod_id in tbcfg["device_to_pod"].items():
        device_id = client.get_device_id(device_name)
        field_map, gate_map = build_sensor_field_map(client.get_attributes(device_id), pod_id=pod_id)
        devices[pod_id] = {"device_id": device_id, "field_map": field_map, "gate_map": gate_map}
        print(f"resolved {device_name} -> {pod_id} ({device_id}), fields: {field_map}, "
              f"hardware-alarm-gated: {gate_map}")

    return client, tbcfg["poll_interval_seconds"], devices, ml_advisor_device_id


def _fill_missing_gate_evidence(client, dev, raw):
    """
    For any gated field whose _active key came back with ZERO points in this
    poll's window (empty, not true/false), look up ThingsBoard's LATEST known
    value for that key -- unbounded by the window -- and splice it into raw
    before parsing. Without this, converters.py's gating has no evidence to
    work with and falls back to "not active" purely because nothing happened
    to transition during this specific poll, which is not the same as actually
    knowing the alarm is inactive.

    If true: still gated out (blocked) same as before.
    If false: now included as normal data, instead of being kept by accident
    via the same fail-open default that also covers "truly never fired".
    If the lookup itself comes back empty too (the alarm has truly never
    fired, ever), it still falls through to converters.py's fail-open default.
    """
    missing_keys = [key for key in dev["gate_map"].values() if not raw.get(key)]
    if not missing_keys:
        return
    try:
        latest = client.get_latest_timeseries(dev["device_id"], missing_keys)
    except Exception as e:
        print(f"failed to look up last known alarm state for {missing_keys}: {e}")
        return
    for key, points in latest.items():
        if points:
            raw[key] = points


def _tb_poll_snapshot(client, devices, last_ts_ms, now_ms):
    """Pull each pod's new readings in (last_ts_ms, now_ms], and build one
    'current state' Snapshot per pod by folding the window's entries together
    (later timestamp overwrites earlier, per field). Returns the Snapshot, or
    None if no pod reported anything.

    NOT just "take the single newest timestamp's dict": each raw sensor field
    updates on its OWN schedule (throttled by its own deadband), so different
    fields land at different timestamps within the same window -- confirmed
    against a real capture (format.1786991988634.json), where taking only the
    latest timestamp discarded every other field that had reported moments
    earlier in the same window. Folding preserves each field's latest value in
    the window, and correctly leaves a field out entirely only when its
    capability's alarm is currently active (see converters._is_currently_active)
    and it has no other in-window point to fall back to.
    """
    from converters import from_thingsboard_timeseries, build_snapshot

    readings_by_pod = {}
    for pod_id, dev in devices.items():
        # field_map keys = the raw sensor_id readings to fetch; gate_map
        # values = the matching alarm "..._active" keys, fetched alongside so
        # from_thingsboard_timeseries can drop a reading while its hardware
        # alarm is currently active. Nothing else in the alarm_* group
        # (_observed etc.) is ever requested.
        keys = list(dev["field_map"].keys()) + list(dev["gate_map"].values())
        if not keys:
            continue
        raw = client.get_timeseries(dev["device_id"], keys, last_ts_ms, now_ms)
        _fill_missing_gate_evidence(client, dev, raw)
        parsed = from_thingsboard_timeseries(raw, dev["field_map"], dev["gate_map"])
        if parsed:
            merged = {}
            for _, fields in parsed:  # chronological -- later overwrites earlier, per field
                merged.update(fields)
            if merged:
                readings_by_pod[pod_id] = merged
    if not readings_by_pod:
        return None
    # Timestamp the snapshot with the poll window's own "now" (now_ms), not
    # wall-clock datetime.now(): they're normally the same instant for live
    # polling, but diverge completely under batch replay (sim-thingsboard runs
    # a whole multi-day window in milliseconds of real time) -- using
    # datetime.now() there collapsed every snapshot onto ~the same timestamp,
    # making collected_span_days() read as ~0 regardless of how many days of
    # ground truth were actually fed through.
    return build_snapshot(datetime.fromtimestamp(now_ms / 1000.0), readings_by_pod)


TEST_OUTPUT_PULSE_MS = 5000  # fixed 5s pulse, not user-configurable (see design notes)
TEST_OUTPUT_MIN_SEVERITY = "critical"  # only the most severe tier gets a
    # physical response; warning/info still push the TB alarm as normal,
    # just without touching any hardware -- Yang's call, not a technical
    # limitation.


def _push_alarms_to_tb(client, devices, ml_advisor_device_id, severity_map, result):
    """
    Push every alert in this poll's result to ThingsBoard as a native Alarm,
    attached to the standalone ML_ADVISOR device -- NEVER to a real Pod's own
    device (see docs: ML advisories must stay physically separate from the
    Pod/Coordinator alarm system mirrored via the TB Rule Chain, to avoid
    mixing hardware-verified alarms with ML inferences on the same entity).

    Any Pods the alert is attributed to (result["alerts"][i]["pod_ids"] -- see
    state.Alert.target_pod_ids()) go into the alarm's `details.contributing_pods`
    instead of choosing which device to attach to, so a single cross-Pod L2
    scenario stays a single alarm record rather than being split one-per-pod.

    Also fires a fixed-duration test_output RPC (LED + buzzer, 5s) at every
    Pod in pod_ids -- but ONLY for the most severe tier (severity ==
    TEST_OUTPUT_MIN_SEVERITY); warning/info still push the TB alarm exactly
    as before, just without any physical trigger. Deliberately ALL pods in
    pod_ids, not just whichever one seems "primary", since that judgment
    call isn't always well-defined for a genuinely cross-Pod scenario and
    reusing pod_ids as-is needed no new design. A fixed short pulse (rather
    than tracking active/silence state the way the Pod's own local alarms
    do) is a deliberate fit for this layer's actual behaviour: there is no
    active/cleared state machine here at all -- every poll where a
    condition still holds just re-pushes the same alert, so a
    self-expiring pulse re-fires on the same rhythm the condition re-fires,
    with nothing to track and nothing to silence.
    Pods whose LED/buzzer wiring hasn't been bench-verified yet (see
    SiteTwin's shared_alarm_indicator_verified) will reject this RPC --
    expected and harmless, silently skipped; the ThingsBoard alarm itself
    is entirely unaffected either way.

    Fires on every poll where the condition still holds; there is no
    de-dup/clearing yet, so an ongoing condition (e.g. a door left open)
    re-pushes each poll until it clears -- a known simplification, revisit if
    ThingsBoard ends up flooded with repeats.
    Best-effort: a push failure is printed, not raised, so one bad push doesn't
    kill the poll loop.
    """
    for alert in result["alerts"]:
        pod_ids = alert["pod_ids"] or ([alert["pod_id"]] if alert["pod_id"] else [])
        tb_severity = severity_map.get(alert["severity"], "INDETERMINATE")
        try:
            client.create_alarm(
                ml_advisor_device_id, alert["rule_id"], tb_severity,
                details={
                    "message": alert["message"],
                    "layer": alert["layer"],
                    "contributing_pods": pod_ids,
                })
        except Exception as e:
            print(f"failed to push alarm {alert['rule_id']} (pods {pod_ids}) to ThingsBoard: {e}")
        if alert["severity"] != TEST_OUTPUT_MIN_SEVERITY:
            continue
        for pod_id in pod_ids:
            dev = devices.get(pod_id)
            if dev is None:
                continue
            for target in ("led", "buzzer"):
                try:
                    client.send_rpc(dev["device_id"], "test_output",
                                    {"target": target, "duration_ms": TEST_OUTPUT_PULSE_MS})
                except Exception as e:
                    # Expected/harmless for a Pod whose LED/buzzer wiring
                    # isn't bench-verified yet -- not worth distinguishing
                    # from other failures here, doesn't affect the alarm
                    # already pushed above.
                    print(f"test_output {target} skipped for {pod_id}: {e}")


_ml_advisor_token_warned = False


def _send_ml_advisor_heartbeat(client, ml_advisor_token):
    """Best-effort: push a heartbeat telemetry point to ML_ADVISOR using its
    own Device Access Token, purely so TB shows it as Active while this
    service is running (mirrors how bridge.py's persistent MQTT connection
    keeps sitetwin-gateway-debug Active -- this is the REST-polling
    equivalent, with the tradeoff of a delay bounded by ML_ADVISOR's Device
    Profile Inactivity Timeout rather than being instant). Silently skipped
    (with one warning) if TB_ML_ADVISOR_TOKEN isn't set -- this is a
    cosmetic feature, not required for alarm pushing to work."""
    global _ml_advisor_token_warned
    if not ml_advisor_token:
        if not _ml_advisor_token_warned:
            print("TB_ML_ADVISOR_TOKEN not set -- ML_ADVISOR will show as "
                  "Inactive in TB (cosmetic only, alarms still work).")
            _ml_advisor_token_warned = True
        return
    try:
        client.push_telemetry_by_token(ml_advisor_token, {"heartbeat": True})
    except Exception as e:
        print(f"failed to push ML_ADVISOR heartbeat: {e}")


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


def _cold_start_thingsboard(config, client, poll_interval, devices, ml_advisor_device_id):
    """
    Collection phase for real data: poll, keep the L2 'confirmed normal'
    snapshots in a disk-backed buffer (L0/L1 are not run -- hardware already
    handles data-quality gatekeeping and hard-limit alerting), and once the
    buffer SPANS collection_days (earliest to latest timestamp, so a restart
    resumes and idle time doesn't count), run diagnostics and train. The buffer
    persists across restarts. Returns once L3 is trained + activated.
    """
    days = config["cold_start"]["collection_days"]
    recheck_interval = config["cold_start"]["recheck_interval_hours"] * 3600
    severity_map = config["thingsboard"].get("severity_map", {})
    l2_pipeline = Pipeline(config, layers=["L2"])  # no model yet
    buffer = _load_buffer()
    print(f"No local model yet -- cold start: collecting normal data until it "
          f"spans {days} days (buffer has {len(buffer)} samples, "
          f"{collected_span_days(buffer):.1f} days so far).")

    ml_advisor_token = os.environ.get("TB_ML_ADVISOR_TOKEN")
    last_ts_ms = int(time.time() * 1000)
    last_train_attempt = 0.0  # wall-clock of the last train/diagnose attempt
    while True:
        now_ms = int(time.time() * 1000)
        _send_ml_advisor_heartbeat(client, ml_advisor_token)
        snap = _tb_poll_snapshot(client, devices, last_ts_ms, now_ms)
        last_ts_ms = now_ms

        if snap is None:
            print(f"[{datetime.fromtimestamp(now_ms/1000.0)}] poll: no data")
        else:
            print(f"[{snap.timestamp}] poll got: {snap.readings}")
            result = l2_pipeline.run(snap)
            # L2 is a real detector even during collection, not just a training
            # filter -- push its hits the same as the steady-state loop does.
            if result["alert_state"] != "normal":
                _push_alarms_to_tb(client, devices, ml_advisor_device_id, severity_map, result)
            else:
                buffer.append(snap)
                _save_buffer(buffer)
                span = collected_span_days(buffer)
                print(f"collected {len(buffer)} normal samples, span {span:.2f}/{days} days")
                # Once the span threshold is met, try to train. If diagnostics
                # fail (a feature never varied), don't retry every poll --
                # re-check only once per recheck_interval, since the result
                # won't change until the missing behavior actually shows up in
                # newly collected data.
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
    client, poll_interval, devices, ml_advisor_device_id = _tb_setup(config)

    if not models_ready(config):
        _cold_start_thingsboard(config, client, poll_interval, devices, ml_advisor_device_id)
        print("-" * 60)

    # L0/L1 skipped -- hardware already handles data-quality gatekeeping and
    # hard-limit alerting; this pipeline only runs L2 (scenario rules) and L3 (LOF).
    pipeline = Pipeline(config, layers=["L2", "L3"])
    severity_map = config["thingsboard"].get("severity_map", {})
    ml_advisor_token = os.environ.get("TB_ML_ADVISOR_TOKEN")
    last_ts_ms = int(time.time() * 1000)  # only pull data from now onward
    print(f"polling ThingsBoard every {poll_interval}s ...")
    while True:
        now_ms = int(time.time() * 1000)
        _send_ml_advisor_heartbeat(client, ml_advisor_token)
        snap = _tb_poll_snapshot(client, devices, last_ts_ms, now_ms)
        last_ts_ms = now_ms
        if snap is not None:
            result = pipeline.run(snap)
            if result["alert_state"] != "normal":
                _print_alert(result)
                _push_alarms_to_tb(client, devices, ml_advisor_device_id, severity_map, result)
        time.sleep(poll_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["sim", "sim-thingsboard", "thingsboard"], default="sim",
                         help="sim = simulated data, direct Snapshots (default); "
                              "sim-thingsboard = same simulated data, but through the real "
                              "ThingsBoard JSON parsing/gating code path (no live instance needed); "
                              "thingsboard = poll real data from a live ThingsBoard instance")
    args = parser.parse_args()

    config = load_config()
    if args.source == "sim":
        run_with_simulated_data(config)
    elif args.source == "sim-thingsboard":
        run_with_simulated_thingsboard_data(config)
    else:
        run_with_thingsboard_data(config)