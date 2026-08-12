"""
Data-source conversion layer. pipeline.run() only understands the unified
format and doesn't care where the data came from.
One conversion function per real data source (MQTT/ThingsBoard REST/serial),
all with the same output format: {"pod_01": {...}, "pod_02": {...}, "pod_03": {...}}

Real field names should follow whatever Vineet/ThingsBoard actually provides;
the field names here are placeholder examples only, adjust when wiring up
real hardware.
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


# ---------- ThingsBoard REST parsing (see thingsboard_api_reference.md) ----------
#
# Three ThingsBoard-specific format quirks handled here (quality_flags handling
# is deliberately NOT done yet -- separate follow-up):
#   1. telemetry values arrive as STRINGS even for numbers -> cast explicitly
#   2. keys are opaque SLOT_<n> slots, not field names -> map via each device's
#      _sensor_kind attribute (which physical sensor sits in which slot is NOT
#      fixed, so it must be looked up, never hardcoded per slot position)
#   3. timestamps are epoch MILLISECONDS, and results are newest-first

# ThingsBoard _sensor_kind attribute value -> our canonical field name.
# temperature_c / relative_humidity_percent / illuminance_lux are the only
# values confirmed in the doc; the rest are best-guess names and MUST be
# checked against the real firmware vocabulary before trusting them.
SENSOR_KIND_TO_FIELD = {
    "temperature_c": "temperature",              # NOTE: env temp; equipment surface temp
                                                  # may also report as temperature_c -- needs a
                                                  # distinct kind or pod-context to disambiguate
    "surface_temperature_c": "equip_temp",       # guessed
    "equipment_temperature_c": "equip_temp",     # guessed
    "relative_humidity_percent": "humidity",
    "co2_ppm": "co2",                             # guessed
    "voc_index": "voc_index",                     # guessed
    "illuminance_lux": "light_lux",
    "current_ma": "current",                      # guessed (unit confirmed mA)
    "vibration_rms_ms2": "vibration_rms",         # guessed (unit confirmed m/s^2)
    "motion": "pir_triggered",                    # guessed
    "pir": "pir_triggered",                       # guessed
    "contact": "door_state",                      # guessed (reed switch)
    "door_state": "door_state",                   # guessed
    "battery_percent": "battery_pct",             # guessed
}

# fields our system treats as booleans (raw telemetry is still a string)
_BOOL_FIELDS = {"pir_triggered", "door_state"}


def build_slot_field_map(attributes_response):
    """
    From one device's CLIENT_SCOPE attributes (list of {key, value, ...}, where
    attribute values come back as native JSON types, not strings), build
    {"SLOT_0": "temperature", "SLOT_1": "humidity", ...}. Slots whose
    _sensor_kind we don't recognize are skipped (returned map just omits them),
    so an unexpected firmware vocabulary won't crash parsing -- it will simply
    drop that slot until SENSOR_KIND_TO_FIELD is extended.
    """
    slot_map = {}
    for attr in attributes_response:
        key = attr.get("key", "")
        if key.endswith("_sensor_kind"):
            slot = key[: -len("_sensor_kind")]  # "SLOT_0_sensor_kind" -> "SLOT_0"
            field = SENSOR_KIND_TO_FIELD.get(attr.get("value"))
            if field is not None:
                slot_map[slot] = field
    return slot_map


def _ms_to_datetime(ts_ms):
    """ThingsBoard ts is epoch milliseconds -> naive local datetime (matches the
    datetime.now()-style timestamps the rest of the pipeline already uses)."""
    return datetime.fromtimestamp(ts_ms / 1000.0)


def _cast_value(field, raw_value):
    """Telemetry values are strings; cast to the type our pipeline expects."""
    if raw_value is None:
        return None
    if field in _BOOL_FIELDS:
        return str(raw_value).strip().lower() in ("1", "true", "on", "open")
    return float(raw_value)


def from_thingsboard_timeseries(timeseries_response, slot_field_map):
    """
    Parse ONE device's (one pod's) timeseries response into a chronologically
    ordered list of (datetime, {field: value}) -- one entry per distinct
    timestamp, holding whatever fields reported at that timestamp.

    timeseries_response: {"SLOT_0": [{"ts": <ms>, "value": "<str>"}, ...],
                          "SLOT_0_quality_flags": [...], "SLOT_1": [...], ...}
    slot_field_map: {"SLOT_0": "temperature", ...} from build_slot_field_map()

    Only the bare SLOT_<n> keys (actual readings) are used; suffixed keys
    (_quality_flags / _sequence / _uptime_ms) are ignored for now. Assembling
    these per-pod streams into full multi-pod Snapshots (aligning pods that
    report at slightly different timestamps) is a separate downstream step,
    not done here.
    """
    by_ts = {}  # ts_ms -> {field: value}
    for slot_key, points in timeseries_response.items():
        field = slot_field_map.get(slot_key)
        if field is None:
            continue  # a suffixed key, or an unmapped slot -- skip
        for point in points:
            ts_ms = point["ts"]
            by_ts.setdefault(ts_ms, {})[field] = _cast_value(field, point.get("value"))

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
