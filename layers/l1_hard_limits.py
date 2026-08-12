"""
L1 hard-limits layer.
Responsibility: unconditional hard thresholds, ignoring all context.
Only processes pods that L0 marked trustworthy (ok).
"""


class HardLimits:
    def __init__(self, config):
        self.cfg = config["hard_limits"]

    def process(self, snapshot):
        for pod_id, reading in snapshot.readings.items():
            # A stuck-suspect signal is still probabilistic, the safety net shouldn't be
            # disabled because of it, see state.is_pod_trustworthy
            if not snapshot.is_pod_trustworthy(pod_id, allow_stuck_suspect=True):
                continue

            temp = reading.get("temperature")
            if temp is not None and temp > self.cfg["temperature_critical"]:
                snapshot.add_alert("L1", "hard_temp_max", "critical",
                                    f"{pod_id} temperature reached {temp}°C, above the absolute limit", pod_id)

            co2 = reading.get("co2")
            if co2 is not None and co2 > self.cfg["co2_critical"]:
                snapshot.add_alert("L1", "hard_co2_max", "critical",
                                    f"{pod_id} CO2 reached {co2}ppm, above the absolute limit", pod_id)

            equip_temp = reading.get("equip_temp")
            if equip_temp is not None and equip_temp > self.cfg["equip_temp_critical"]:
                snapshot.add_alert("L1", "hard_equip_temp_max", "critical",
                                    f"{pod_id} equipment temperature reached {equip_temp}°C", pod_id)

            vib = reading.get("vibration_rms")
            if vib is not None and vib > self.cfg["vib_abnormal_rms"]:
                snapshot.add_alert("L1", "hard_vibration", "warning",
                                    f"{pod_id} vibration RMS reached {vib}, above baseline", pod_id)
        return snapshot
