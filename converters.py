"""
Data-source conversion layer. pipeline.run() only understands the unified
format and doesn't care where the data came from.
One conversion function per real data source (MQTT/ThingsBoard REST/serial),
all with the same output format: {"pod_01": {...}, "pod_02": {...}, "pod_03": {...}}
"""
from datetime import datetime
from state import Snapshot


def from_mqtt_message(topic, payload_json):
    """
    A single MQTT message usually carries data for one pod only.
    The caller must either assemble the three pods' data within the same time
    window before building a Snapshot, or accept "a snapshot only contains
    whichever pod happened to report" as an incomplete snapshot
    (L0/L2 need to handle missing-pod fields as None).
    """
    pod_id = topic.split("/")[1]  # e.g. sitetwin/pod1/telemetry -> pod1
    return pod_id, payload_json.get("sensors", {})


# ---------- ThingsBoard REST parsing (see TB_Data_Reference_for_ML.md, captured
# live from real hardware 2026-08-16 -- ground truth, supersedes the older
# thingsboard_api_reference.md's SLOT_<n> description, which real devices don't
# actually use) ----------
#
# Real format, four things to handle:
#   1. telemetry values arrive as STRINGS even for numbers/booleans -> cast explicitly
#   2. reading keys are real semantic sensor ids (e.g. "sht41_temperature",
#      "scd41_co2"), not opaque slots -- but which physical sensor is present on
#      a given device is still only knowable via that device's own
#      "{sensor_id}_sensor_kind" attribute, so it must be looked up per device,
#      never hardcoded by key name alone (two different sensor ids can report
#      the same sensor_kind -- see _POD_SPECIFIC_KIND_TO_FIELD below)
#   3. timestamps are epoch MILLISECONDS, and results are newest-first
#   4. values always come from the raw sensor_id field, never from the
#      alarm_{capability}_{rule_kind}_* group -- the ONLY thing polled from
#      that group is "..._active", and only to gate the raw reading: when a
#      capability's alarm is currently active, its reading is dropped entirely
#      for this poll (hardware already considers it out of bounds, doc 4.3 --
#      we don't want L2/L3 treating it as a normal sample). Everything else in
#      that group (_observed/_instance_id/_threshold/_acknowledged/_transition/
#      _reason/_sensor_id/_quality_flags) is never requested.

# ThingsBoard "sensor_kind" attribute value -> our canonical field name.
# Confirmed against real captured data (TB_Data_Reference_for_ML.md section 6)
# unless marked "unconfirmed".
SENSOR_KIND_TO_FIELD = {
    "temperature_c": "temperature",               # confirmed (env pod's sht41_temperature);
                                                    # equipment pod's ds18b20_temperature reports
                                                    # this SAME sensor_kind string -- see
                                                    # _POD_SPECIFIC_KIND_TO_FIELD, which overrides
                                                    # this default on the equipment pod
    "relative_humidity_percent": "humidity",       # confirmed
    "co2_ppm": "co2",                              # confirmed
    "voc_index": "voc_index",                      # confirmed
    "illuminance_lux": "light_lux",                # confirmed
    "current_ma": "current",                       # confirmed key name (unit mA)
    "vibration_rms_g": "vibration_rms",            # confirmed (real capability string; unit is g,
                                                    # not m/s^2 as previously assumed -- see
                                                    # hard_limits.vib_abnormal_rms and l2_context.py,
                                                    # updated to match. The threshold NUMBERS there
                                                    # are still placeholders pending real g-scale
                                                    # calibration -- the unit label is fixed, the
                                                    # values are not yet)
    "motion": "pir_triggered",                     # confirmed (pir_motion sensor)
    "contact": "door_state",                       # confirmed (reed_contact sensor)
    "battery_percent": "battery_pct",              # not seen in real captured data yet, still a guess
}

# Some sensor_kind strings are ambiguous across pods -- the SAME kind is
# reported by physically different sensors depending on which pod the device
# belongs to (e.g. "temperature_c" is both the environment pod's ambient
# sensor and the equipment pod's surface sensor). Looked up by pod_id first;
# falls back to SENSOR_KIND_TO_FIELD when there's no pod-specific override.
_POD_SPECIFIC_KIND_TO_FIELD = {
    "pod_01": {"temperature_c": "temperature"},   # environment pod: sht41_temperature
    "pod_03": {"temperature_c": "equip_temp"},    # equipment pod: ds18b20_temperature
}

# fields our system treats as booleans (raw telemetry is still a string)
_BOOL_FIELDS = {"pir_triggered", "door_state"}


def _discover_rule_kinds(attributes_response):
    """
    From "rule_{capability}_{rule_kind}_value" attributes (e.g.
    "rule_temperature_c_numeric_high_threshold_value"), build
    {capability: rule_kind} for every capability that has a threshold rule
    configured on this device. capability is matched against the known
    sensor_kind vocabulary (SENSOR_KIND_TO_FIELD / _POD_SPECIFIC_KIND_TO_FIELD)
    since capability names can themselves contain underscores.
    """
    known_capabilities = set(SENSOR_KIND_TO_FIELD)
    for overrides in _POD_SPECIFIC_KIND_TO_FIELD.values():
        known_capabilities.update(overrides)

    rule_kinds = {}
    for attr in attributes_response:
        key = attr.get("key", "")
        if not (key.startswith("rule_") and key.endswith("_value")):
            continue
        middle = key[len("rule_"): -len("_value")]  # "temperature_c_numeric_high_threshold"
        for cap in known_capabilities:
            prefix = cap + "_"
            if middle.startswith(prefix):
                rule_kinds[cap] = middle[len(prefix):]
                break
    return rule_kinds


def build_sensor_field_map(attributes_response, pod_id=None):
    """
    From one device's CLIENT_SCOPE attributes (list of {key, value, ...}, where
    attribute values come back as native JSON types, not strings), build
    (field_map, gate_map):

      field_map: {sensor_id: our_field_name} -- the raw reading key for each
        field. Always the raw sensor_id, never an alarm_*_observed key -- of
        the whole alarm_{capability}_{rule_kind}_* field group (doc 8.2:
        _observed/_instance_id/_threshold/_acknowledged/_transition/_reason/
        _sensor_id/_quality_flags), the ONLY one ever polled is _active, and
        only for gating (see gate_map below); none of the others are read.

      gate_map: {our_field_name: alarm_active_telemetry_key} -- for fields
        whose capability has a threshold rule configured, the "..._active" key
        to also poll. from_thingsboard_timeseries drops that field's reading
        whenever this is currently "true" (hardware already flags the value
        out of bounds, see doc 4.3 -- we don't want L2/L3 ingesting it as a
        normal sample).

    pod_id (our "pod_01"/"pod_02"/"pod_03") resolves sensor_kind values that
    mean different things on different pods (see _POD_SPECIFIC_KIND_TO_FIELD);
    pass it whenever known.

    Sensor ids whose sensor_kind we don't recognize are skipped (the returned
    maps just omit them), so an unexpected firmware vocabulary won't crash
    parsing -- it will simply drop that reading until SENSOR_KIND_TO_FIELD is
    extended.
    """
    field_map = {}
    gate_map = {}
    pod_override = _POD_SPECIFIC_KIND_TO_FIELD.get(pod_id, {})
    rule_kinds = _discover_rule_kinds(attributes_response)
    for attr in attributes_response:
        key = attr.get("key", "")
        if not key.endswith("_sensor_kind"):
            continue
        sensor_id = key[: -len("_sensor_kind")]  # "sht41_temperature_sensor_kind" -> "sht41_temperature"
        kind = attr.get("value")
        field = pod_override.get(kind, SENSOR_KIND_TO_FIELD.get(kind))
        if field is None:
            continue
        field_map[sensor_id] = field  # value always comes from the raw field
        rule_kind = rule_kinds.get(kind)
        if rule_kind:
            gate_map[field] = f"alarm_{kind}_{rule_kind}_active"
    return field_map, gate_map


def _ms_to_datetime(ts_ms):
    """ThingsBoard ts is epoch milliseconds -> naive local datetime (matches the
    datetime.now()-style timestamps the rest of the pipeline already uses)."""
    return datetime.fromtimestamp(ts_ms / 1000.0)


def _cast_value(field, raw_value):
    """Telemetry values are strings; cast to the type our pipeline expects."""
    if raw_value is None:
        return None
    if field in _BOOL_FIELDS:
        s = str(raw_value).strip().lower()
        if s in ("true", "1", "on", "open"):
            return True
        if s in ("false", "0", "off", "closed"):
            return False
        # real raw sensor fields for binary sensors encode state as a numeric
        # string, not "true"/"false" -- confirmed real values: pir_motion
        # reports "1.0"/"0.0", reed_contact reports "0.0" (pod2.json). Only the
        # alarm _active/_acknowledged fields use literal "true"/"false".
        try:
            return float(s) != 0.0
        except ValueError:
            return False
    return float(raw_value)


def _is_currently_active(active_points):
    """
    Whether an alarm is active RIGHT NOW, per the most recent "_active"
    transition point in the pulled window. Deliberately NOT "was it active at
    each individual reading's own timestamp": a reading taken before the
    alarm's most recent active transition is stale evidence once we know the
    alarm has since gone active -- reporting it as the "current" value would
    smuggle an about-to-be-flagged (or already flagged) reading through just
    because it happened to be sampled a moment earlier. Empty/no evidence in
    this window -> False (fail-open: keep the reading rather than drop it when
    we can't actually tell).
    """
    if not active_points:
        return False
    latest = max(active_points, key=lambda p: p["ts"])
    return str(latest.get("value")).strip().lower() == "true"


def from_thingsboard_timeseries(timeseries_response, field_map, gate_map=None):
    """
    Parse ONE device's (one pod's) timeseries response into a chronologically
    ordered list of (datetime, {field: value}) -- one entry per distinct
    timestamp, holding whatever fields reported at that timestamp.

    timeseries_response: {"sht41_temperature": [{"ts": <ms>, "value": "<str>"}, ...],
                          "sht41_temperature_quality_flags": [...],
                          "alarm_co2_ppm_numeric_high_threshold_active": [...], ...}
    field_map, gate_map: from build_sensor_field_map()

    Only the keys listed in field_map (the raw sensor_id readings) are used;
    every suffixed key (_quality_flags/_sequence/_uptime_ms, rule_* keys, and
    the alarm_* group other than _active) is ignored. gate_map's _active keys
    are read but never themselves stored as a field -- they only decide
    whether their field is dropped from EVERY entry in this window (see
    _is_currently_active). Assembling these per-pod streams into full
    multi-pod Snapshots (aligning pods that report at slightly different
    timestamps) is a separate downstream step, not done here.
    """
    gate_map = gate_map or {}
    by_ts = {}  # ts_ms -> {field: value}
    for key, points in timeseries_response.items():
        field = field_map.get(key)
        if field is None:
            continue  # a suffixed/alarm/rule/active key, or an unmapped sensor id -- skip
        for point in points:
            ts_ms = point["ts"]
            by_ts.setdefault(ts_ms, {})[field] = _cast_value(field, point.get("value"))

    # Hardware alarm gating: if a field's capability is CURRENTLY active (per
    # the most recent _active transition in this window), drop that field from
    # every entry in this window -- not just entries at-or-after the
    # transition. A reading taken before the transition is stale once we know
    # the alarm has since gone active; letting it through as "the current
    # value" would defeat the point of gating (see _is_currently_active).
    for field, active_key in gate_map.items():
        if _is_currently_active(timeseries_response.get(active_key, [])):
            for fields in by_ts.values():
                fields.pop(field, None)

    # response is newest-first; return chronological (oldest-first)
    return [(_ms_to_datetime(ts_ms), by_ts[ts_ms]) for ts_ms in sorted(by_ts)]


def from_serial_line(line):
    """
    Bare-bones fallback: one CSV-formatted line of serial output
    e.g. pod_01,23.4,58.2,900,80
    """
    parts = line.strip().split(",")
    pod_id = parts[0]
    return pod_id, {
        "temperature": float(parts[1]),
        "humidity": float(parts[2]),
        "co2": float(parts[3]),
        "voc_index": float(parts[4]),
    }


def build_snapshot(timestamp, readings_by_pod):
    return Snapshot(timestamp=timestamp, readings=readings_by_pod)
