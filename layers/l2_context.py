"""
L2 context-rules layer.
Two steps:
  1. discretize()  raw sensor values -> labels
  2. ScenarioRules  combined-label scenario rules

Design history (see SiteTwin_L0-L3架构说明.docx sections 七/八 for the full argument):
  This layer used to maintain an occupied/unoccupied/uncertain state machine
  driven by PIR + door events + CO2 trend, and most scenario rules were gated
  on that inferred state. That state machine was removed entirely: PIR only
  detects motion (not presence, so silence != empty), door events require an
  unreliable state->event conversion at the data source, and CO2 in a real
  multi-equipment environment can't be reliably attributed to human
  respiration. No combination of these three signals can answer "is someone
  in the room right now" with the confidence needed to gate an alert.

  Every scenario rule below only uses signals that are either (a) directly
  and continuously observable (e.g. current door open/closed state, not an
  inferred occupancy label) or (b) cross-pod correlations where independent
  physical channels agreeing is itself the evidence, rather than any single
  channel's silence being read as "nobody's here".
"""
from datetime import timedelta


def discretize(config, pod_id, reading):
    """Convert one pod's raw reading into a label dict"""
    lc = config["labels"]
    labels = {}

    temp = reading.get("temperature")
    if temp is not None:
        if temp < lc["temperature"]["low"]:
            labels["temp_label"] = "low"
        elif temp < lc["temperature"]["high"]:
            labels["temp_label"] = "normal"
        elif temp <= config["hard_limits"]["temperature_critical"]:
            labels["temp_label"] = "high"
        else:
            labels["temp_label"] = "critical"

    humidity = reading.get("humidity")
    if humidity is not None:
        if humidity < lc["humidity"]["low"]:
            labels["humidity_label"] = "low"
        elif humidity <= lc["humidity"]["high"]:
            labels["humidity_label"] = "normal"
        else:
            labels["humidity_label"] = "high"

    co2 = reading.get("co2")
    if co2 is not None:
        if co2 < lc["co2"]["normal"]:
            labels["co2_label"] = "normal"
        elif co2 < lc["co2"]["elevated"]:
            labels["co2_label"] = "elevated"
        elif co2 <= config["hard_limits"]["co2_critical"]:
            labels["co2_label"] = "high"
        else:
            labels["co2_label"] = "critical"

    voc = reading.get("voc_index")
    if voc is not None:
        if voc < lc["voc"]["elevated"]:
            labels["voc_label"] = "normal"
        elif voc < lc["voc"]["high"]:
            labels["voc_label"] = "elevated"
        else:
            labels["voc_label"] = "high"

    if "pir_triggered" in reading:
        labels["motion_label"] = "motion_recent" if reading["pir_triggered"] else "motion_none"

    if "door_state" in reading:
        # continuous open/closed reading straight from the reed switch, not an
        # edge-triggered event -- no state->event conversion needed for this
        labels["door_label"] = "open" if reading["door_state"] else "closed"

    if "light_lux" in reading:
        labels["light_label"] = "light_on" if reading["light_lux"] >= lc["light_lux_on"] else "light_off"

    if "current" in reading:
        # unit confirmed as mA. Banding is still a placeholder (small-motor magnitude,
        # not calibrated against real hardware), needs recalibration once real current
        # draw data is available.
        current = reading["current"]
        if current is None or current < 10:
            labels["current_label"] = "off"
        elif current < 400:
            labels["current_label"] = "normal"
        else:
            labels["current_label"] = "high"

    if "vibration_rms" in reading:
        # unit is g, not m/s^2 as previously assumed -- corrected per
        # TB_Data_Reference_for_ML.md (real capability name is vibration_rms_g).
        # Delivered pre-computed by the sensor/firmware from the X/Y/Z axes
        # (gravity-removed, per the standard vibration-RMS convention -- baseline
        # near 0 at rest, not near 1g). Both the 0.02 "none" cutoff below and the
        # "normal"/"abnormal" band threshold (hard_limits.vib_abnormal_rms) were
        # only ever placeholders and are still pending real g-scale calibration
        # against the ADXL345 -- the unit fix here does not make these numbers
        # correct, just correctly labeled.
        vib = reading["vibration_rms"]
        if vib is None or vib < 0.02:
            labels["vib_label"] = "none"
        elif vib < config["hard_limits"]["vib_abnormal_rms"]:
            labels["vib_label"] = "normal"
        else:
            labels["vib_label"] = "abnormal"

    return labels


class ScenarioRules:
    """
    L2 combined scenario rules. Every rule here is either a single-pod
    instantaneous check (equip_high_load / equip_stall_risk /
    vibration_mechanical_fault / door_left_open) or a cross-pod trend
    correlation (equip_overheat_air_quality / equip_cooling_failure /
    rapid_multi_signal_spike). None of them read an occupancy label.
    """
    def __init__(self, config):
        self.cfg = config["scenarios"]
        self._door_open_since = {}       # pod_id -> timestamp, for door_left_open
        self._equip_temp_history = {}    # pod_id -> [(ts, value), ...]
        self._current_history = {}       # pod_id -> [(ts, value), ...]
        self._temperature_history = {}   # pod_id -> [(ts, value), ...] (env pod)
        self._vibration_history = {}     # pod_id -> [(ts, value), ...]

    def process(self, snapshot, env_pod="pod_01", equip_pod="pod_03"):
        for pod_id, labels in snapshot.sensor_labels.items():
            if not snapshot.is_pod_trustworthy(pod_id):
                continue
            self._check_equipment_instant(snapshot, pod_id, labels)
            self._check_door_left_open(snapshot, pod_id, labels)

        self._update_histories(snapshot, env_pod, equip_pod)
        self._check_equip_thermal_trends(snapshot, env_pod, equip_pod)
        self._check_rapid_multi_signal_spike(snapshot, env_pod, equip_pod)
        return snapshot

    # ---------- single-pod instantaneous checks ----------

    def _check_equipment_instant(self, snapshot, pod_id, labels):
        current_label = labels.get("current_label")
        vib_label = labels.get("vib_label")
        if current_label == "high" and vib_label == "normal":
            snapshot.add_alert("L2", "scene_equip_high_load",
                                self.cfg["equip_high_load"]["severity"],
                                f"{pod_id} equipment running under high load", pod_id)
        if current_label == "high" and vib_label == "none":
            snapshot.add_alert("L2", "scene_equip_stall_risk",
                                self.cfg["equip_stall_risk"]["severity"],
                                f"{pod_id} current elevated but no vibration, possible stall", pod_id)
        if vib_label == "abnormal":
            snapshot.add_alert("L2", "scene_vibration_mechanical_fault",
                                self.cfg["vibration_mechanical_fault"]["severity"],
                                f"{pod_id} vibration abnormal, "
                                f"possible mechanical looseness or bearing fault", pod_id)

    def _check_door_left_open(self, snapshot, pod_id, labels):
        door_label = labels.get("door_label")
        now = snapshot.timestamp
        if door_label == "open":
            since = self._door_open_since.setdefault(pod_id, now)
            cfg = self.cfg["door_left_open"]
            if (now - since).total_seconds() >= cfg["duration_minutes"] * 60:
                snapshot.add_alert("L2", "scene_door_left_open", cfg["severity"],
                                    f"{pod_id} door has been open for a long time", pod_id)
        else:
            self._door_open_since.pop(pod_id, None)

    # ---------- cross-pod trend correlations ----------

    def _update_histories(self, snapshot, env_pod, equip_pod):
        now = snapshot.timestamp
        env = snapshot.readings.get(env_pod, {})
        equip = snapshot.readings.get(equip_pod, {})
        max_window = max(
            self.cfg["equip_overheat_air_quality"]["window_minutes"],
            self.cfg["equip_cooling_failure"]["window_minutes"],
            self.cfg["rapid_multi_signal_spike"]["window_minutes"],
        )
        window = timedelta(minutes=max_window)

        def _append(hist_dict, pod_id, value):
            if value is None:
                return
            hist = hist_dict.setdefault(pod_id, [])
            hist.append((now, value))
            hist_dict[pod_id] = [(t, v) for t, v in hist if now - t <= window]

        _append(self._equip_temp_history, equip_pod, equip.get("equip_temp"))
        _append(self._current_history, equip_pod, equip.get("current"))
        _append(self._temperature_history, env_pod, env.get("temperature"))
        _append(self._vibration_history, equip_pod, equip.get("vibration_rms"))

    @staticmethod
    def _trend(history, pod_id, now, window_minutes):
        """Return (oldest_value, newest_value, delta) within the window, or None."""
        window = timedelta(minutes=window_minutes)
        pts = [(t, v) for t, v in history.get(pod_id, []) if now - t <= window]
        if len(pts) < 2:
            return None
        pts.sort(key=lambda x: x[0])
        oldest, newest = pts[0][1], pts[-1][1]
        return oldest, newest, newest - oldest

    def _check_equip_thermal_trends(self, snapshot, env_pod, equip_pod):
        if not snapshot.is_pod_trustworthy(equip_pod):
            return
        now = snapshot.timestamp

        # scenario: equipment overheating with corroborating air-quality change (fire precursor)
        oa_cfg = self.cfg["equip_overheat_air_quality"]
        temp_trend = self._trend(self._equip_temp_history, equip_pod, now, oa_cfg["window_minutes"])
        current_trend = self._trend(self._current_history, equip_pod, now, oa_cfg["window_minutes"])
        voc_label = snapshot.sensor_labels.get(env_pod, {}).get("voc_label")

        equip_temp_rising = temp_trend is not None and temp_trend[2] >= oa_cfg["equip_temp_rise_c"]
        current_rising = (current_trend is not None and current_trend[0]
                           and current_trend[2] / current_trend[0] >= oa_cfg["current_rise_ratio"])

        if equip_temp_rising and current_rising and voc_label in ("elevated", "high"):
            snapshot.add_alert(
                "L2", "scene_equip_overheat_air_quality", oa_cfg["severity"],
                f"{equip_pod} equipment temperature and current rising together with elevated VOC, "
                f"possible overheating", equip_pod)

        # scenario: equipment temperature rising without a matching current rise (cooling failure)
        cf_cfg = self.cfg["equip_cooling_failure"]
        temp_trend2 = self._trend(self._equip_temp_history, equip_pod, now, cf_cfg["window_minutes"])
        current_trend2 = self._trend(self._current_history, equip_pod, now, cf_cfg["window_minutes"])
        equip_temp_rising2 = temp_trend2 is not None and temp_trend2[2] >= cf_cfg["equip_temp_rise_c"]
        current_stable = (current_trend2 is not None and current_trend2[0]
                           and abs(current_trend2[2] / current_trend2[0]) <= cf_cfg["current_max_change_ratio"])
        if equip_temp_rising2 and current_stable:
            snapshot.add_alert(
                "L2", "scene_equip_cooling_failure", cf_cfg["severity"],
                f"{equip_pod} temperature rising steadily while current stays stable, "
                f"possible cooling failure", equip_pod)

    def _check_rapid_multi_signal_spike(self, snapshot, env_pod, equip_pod):
        if not (snapshot.is_pod_trustworthy(env_pod) and snapshot.is_pod_trustworthy(equip_pod)):
            return
        now = snapshot.timestamp
        cfg = self.cfg["rapid_multi_signal_spike"]
        w = cfg["window_minutes"]

        temp_trend = self._trend(self._temperature_history, env_pod, now, w)
        equip_temp_trend = self._trend(self._equip_temp_history, equip_pod, now, w)
        vib_trend = self._trend(self._vibration_history, equip_pod, now, w)

        temp_jump = temp_trend is not None and abs(temp_trend[2]) >= cfg["temperature_jump_c"]
        equip_temp_jump = equip_temp_trend is not None and abs(equip_temp_trend[2]) >= cfg["equip_temp_jump_c"]
        vib_jump = vib_trend is not None and abs(vib_trend[2]) >= cfg["vibration_jump"]

        if temp_jump and equip_temp_jump and vib_jump:
            snapshot.add_alert(
                "L2", "scene_rapid_multi_signal_spike", cfg["severity"],
                "Multiple independent signals (ambient temperature, equipment temperature, vibration) "
                "spiked within a short window, possible violent event",
                pod_ids=[env_pod, equip_pod])


class ContextLayer:
    """L2 entry point: discretize -> scenario rules"""
    def __init__(self, config):
        self.config = config
        self.scenario_rules = ScenarioRules(config)

    def process(self, snapshot):
        for pod_id, reading in snapshot.readings.items():
            if not snapshot.is_pod_trustworthy(pod_id):
                continue
            snapshot.sensor_labels[pod_id] = discretize(self.config, pod_id, reading)

        self.scenario_rules.process(snapshot)
        return snapshot
