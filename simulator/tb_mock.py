"""
Mock ThingsBoard REST client, backed by synthetic ground-truth data instead of
a live server. Drop-in replacement for thingsboard_client.ThingsBoardClient
(same get_device_id/get_attributes/get_timeseries shapes), so main.py's real
polling code (_tb_poll_snapshot) and converters.py's real parsing/gating
(build_sensor_field_map / from_thingsboard_timeseries) run completely
unchanged against it -- this exercises the REAL code path end-to-end without a
live instance, using the real per-pod device schema confirmed against actual
hardware captures (TB_Data_Reference_for_ML.md, pod1.json/pod2.json/pod3.json).

Ground truth comes from simulator.generate.generate_day() -- the same
Snapshots --source sim uses -- so the two simulated modes agree on "what
happened"; only the wire format and the parsing code path differ.
"""

# Per-pod sensor schema: (sensor_id, capability, our_field, rule_kind, threshold).
# rule_kind/threshold are None for sensors with no configured alarm rule
# (matches pir_motion in the real capture -- "event-type, no alarm fields").
#
# sensor_id/capability/rule_kind are the REAL ones captured from hardware.
# Thresholds are NOT all copied literally from the real capture, though:
# mock data is meant to represent normal operation (this mode collects/trains
# on it, it isn't the anomaly-injection tool -- that's evaluate.py's job), so a
# threshold has to actually sit above our simulator's normal range or every
# single reading gets gated as "hardware already flagged it", which is what
# happened testing this against the real captured current(-1.0)/vibration(0.0)
# thresholds: both trip on ~100% of normal samples (any current >= 0mA or
# vibration >= 0g trips them -- those read as unconfigured/placeholder values
# on the real device, not real calibration). humidity(30.0) has the same
# problem for a different reason: our OWN config.yaml labels 30-60% as the
# *normal* band, so a "high" alarm AT 30 would flag all of normal operation.
# temperature/co2/voc/illuminance/contact's real thresholds don't have this
# problem (confirmed empirically -- normal simulated data rarely/never crosses
# them) and are kept as captured. The three replaced below reuse OUR OWN
# already-defined normal/abnormal boundaries instead (current_label's "high"
# cutoff in l2_context.py, hard_limits.vib_abnormal_rms, labels.humidity.high)
# so mock alarm behavior is at least internally consistent with the rest of
# the system, pending real recalibration on the hardware side.
POD_SCHEMA = {
    "pod_01": [
        ("sht41_temperature", "temperature_c", "temperature", "numeric_high_threshold", 30.0),
        ("sht41_humidity", "relative_humidity_percent", "humidity", "numeric_high_threshold", 60.0),  # was real 30.0 -- see note above
        ("scd41_co2", "co2_ppm", "co2", "numeric_high_threshold", 1000.0),
        ("sgp40_voc", "voc_index", "voc_index", "numeric_high_threshold", 80.0),
    ],
    "pod_02": [
        ("bh1750_illuminance", "illuminance_lux", "light_lux", "numeric_high_threshold", 800.0),
        ("pir_motion", "motion", "pir_triggered", None, None),
        ("reed_contact", "contact", "door_state", "contact_state_active_value", 1.0),
    ],
    "pod_03": [
        ("ds18b20_temperature", "temperature_c", "equip_temp", "numeric_high_threshold", 30.0),
        ("ina219_current", "current_ma", "current", "numeric_high_threshold", 400.0),          # was real -1.0 -- see note above
        ("adxl345_vibration", "vibration_rms_g", "vibration_rms", "numeric_high_threshold", 1.0),  # was real 0.0 -- see note above
    ],
}

# fields our system treats as booleans -- real encoding confirmed in pod2.json
# is a numeric string ("1.0"/"0.0"), not "true"/"false"
_BOOL_FIELDS = {"pir_triggered", "door_state"}


def mock_attributes(pod_id):
    """Fake CLIENT_SCOPE attributes response for one pod's device: one
    {sensor_id}_sensor_kind attribute per sensor, plus one
    rule_{capability}_{rule_kind}_value attribute for each sensor that has a
    configured rule -- exactly what build_sensor_field_map() and
    converters._discover_rule_kinds() read."""
    attrs = []
    for sensor_id, capability, _field, rule_kind, threshold in POD_SCHEMA[pod_id]:
        attrs.append({"key": f"{sensor_id}_sensor_kind", "value": capability})
        if rule_kind is not None:
            attrs.append({"key": f"rule_{capability}_{rule_kind}_value", "value": threshold})
    return attrs


def _encode(field, value):
    if value is None:
        return None
    if field in _BOOL_FIELDS:
        return "1.0" if value else "0.0"
    return str(value)


def mock_timeseries_point(pod_id, reading, ts_ms):
    """Fake timeseries response for ONE snapshot's reading on one pod: the raw
    sensor_id field for every value present, plus the
    alarm_{capability}_{rule_kind}_active key (only) for sensors that have a
    rule -- matches what main.py actually requests (build_sensor_field_map only
    asks for the raw field + _active, never _observed or anything else, see
    converters.py)."""
    response = {}
    for sensor_id, capability, field, rule_kind, threshold in POD_SCHEMA[pod_id]:
        value = reading.get(field)
        if value is None:
            continue
        response[sensor_id] = [{"ts": ts_ms, "value": _encode(field, value)}]
        if rule_kind is not None:
            active = value >= threshold
            response[f"alarm_{capability}_{rule_kind}_active"] = [
                {"ts": ts_ms, "value": "true" if active else "false"}
            ]
    return response


class MockThingsBoardClient:
    """
    snapshots: list of state.Snapshot (e.g. from generate_day()) -- the ground
    truth this mock serves. device_id and pod_id are treated as the same thing
    here (no real UUID resolution needed for a mock).
    """

    def __init__(self, snapshots):
        self._by_pod = {}  # pod_id -> sorted [(ts_ms, reading), ...]
        for snap in snapshots:
            ts_ms = int(snap.timestamp.timestamp() * 1000)
            for pod_id, reading in snap.readings.items():
                self._by_pod.setdefault(pod_id, []).append((ts_ms, reading))
        for points in self._by_pod.values():
            points.sort(key=lambda p: p[0])

    def login(self, username, password):
        return "mock-token"

    def get_device_id(self, device_name):
        return device_name  # identity -- device_name IS the pod_id here

    def get_attributes(self, device_id):
        return mock_attributes(device_id)

    def get_timeseries(self, device_id, keys, start_ts, end_ts):
        pod_id = device_id
        wanted = set(keys)
        merged = {}
        for ts_ms, reading in self._by_pod.get(pod_id, []):
            if not (start_ts < ts_ms <= end_ts):
                continue
            for key, points in mock_timeseries_point(pod_id, reading, ts_ms).items():
                if key in wanted:
                    merged.setdefault(key, []).extend(points)
        return merged

    def get_latest_timeseries(self, device_id, keys):
        """Interface parity with ThingsBoardClient (main._fill_missing_gate_evidence
        calls this as a fallback). In practice every mocked reading with a rule
        always carries its _active point alongside it (see mock_timeseries_point),
        so a poll window is never missing gate evidence in the first place and
        this is never actually invoked -- implemented anyway so nothing breaks
        if that ever changes."""
        pod_id = device_id
        wanted = set(keys)
        latest = {}
        for ts_ms, reading in self._by_pod.get(pod_id, []):
            for key, points in mock_timeseries_point(pod_id, reading, ts_ms).items():
                if key in wanted:
                    latest[key] = points  # later ts overwrites earlier (list is time-sorted)
        return latest
