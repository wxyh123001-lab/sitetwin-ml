"""
L0 data gatekeeper layer.
Responsibility: decide whether each pod's data can be trusted, for every
downstream layer to rely on.
Makes no "anomaly" judgement of its own, only a "can this data be believed" judgement.
"""
from datetime import datetime, timedelta


class Gatekeeper:
    def __init__(self, config, node_memory):
        self.cfg = config["gatekeeper"]
        # node_memory: tracks each pod's last-seen time and recent reading history, used for stuck detection
        self.node_memory = node_memory

    def process(self, snapshot):
        for pod_id, reading in snapshot.readings.items():
            status = self._check_pod(pod_id, reading, snapshot.timestamp)
            snapshot.data_quality[pod_id] = status
            if status == "offline":
                snapshot.add_alert("L0", "node_offline", "warning",
                                    f"{pod_id} has not reported data within the timeout", pod_id)
            elif status == "low_battery":
                # low battery doesn't affect data trustworthiness, log an info alert but still treat data as ok downstream
                snapshot.data_quality[pod_id] = "ok"
                snapshot.add_alert("L0", "node_low_battery", "info",
                                    f"{pod_id} battery level is low", pod_id)
            elif status == "sensor_fault":
                snapshot.add_alert("L0", "sensor_fault", "warning",
                                    f"{pod_id} reading is outside the physically possible range", pod_id)
            elif status == "stuck_suspect":
                # Probabilistic suspicion signal (not a physical impossibility), kept trustworthy for
                # the L1 safety net; only L2/L3 skip it, see is_pod_trustworthy(allow_stuck_suspect=...)
                snapshot.add_alert("L0", "node_stuck_suspect", "info",
                                    f"{pod_id} reading hasn't changed for a while, possibly stuck", pod_id)
        return snapshot

    def _check_pod(self, pod_id, reading, now):
        mem = self.node_memory.setdefault(pod_id, {"last_seen": now, "history": []})

        # 1. Offline check
        timeout = self.cfg["offline_timeout_seconds"].get(pod_id, 600)
        if (now - mem["last_seen"]).total_seconds() > timeout and mem["history"]:
            return "offline"
        mem["last_seen"] = now

        # 2. Battery check
        battery = reading.get("battery_pct")
        if battery is not None and battery < self.cfg["low_battery_pct"]:
            return "low_battery"

        # 3. Physical range check
        for key, (lo, hi) in self.cfg["physical_range"].items():
            val = reading.get(key)
            if val is not None and not (lo <= val <= hi):
                return "sensor_fault"

        # 4. Stuck check (reading completely unchanged for a long time) -- probabilistic
        #    suspicion, not a hard fault, so it must not be treated the same as a
        #    physical-range violation (see the stuck_suspect branch above)
        mem["history"].append((now, dict(reading)))
        window = timedelta(minutes=self.cfg["stuck_check"]["window_minutes"])
        mem["history"] = [(t, r) for t, r in mem["history"] if now - t <= window]
        if len(mem["history"]) >= 5:
            for key, min_var in self.cfg["stuck_check"]["min_variation"].items():
                vals = [r.get(key) for _, r in mem["history"] if r.get(key) is not None]
                if len(vals) >= 5 and (max(vals) - min(vals)) < min_var:
                    return "stuck_suspect"

        return "ok"
