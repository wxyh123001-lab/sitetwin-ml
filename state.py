"""
Core data structures. Every layer reads and writes the same Snapshot object.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Alert:
    layer: str          # "L1" / "L2" / "L3"
    rule_id: str         # e.g. "hard_temp_max" / "scene1_unoccupied_temp"
    severity: str         # "info" / "warning" / "critical"
    message: str
    pod_id: Optional[str] = None


@dataclass
class Snapshot:
    """Full system state at one point in time, flowing through L0 -> L1 -> L2 -> L3 -> fusion"""
    timestamp: datetime
    # Raw readings per pod: {"pod_01": {"temperature": 23.4, "co2": 900, ...}, ...}
    readings: dict = field(default_factory=dict)
    # Written by L0: {"pod_01": "ok", "pod_03": "offline", ...}
    data_quality: dict = field(default_factory=dict)
    room_state: dict = field(default_factory=dict)
    # Discretized single-sensor labels: {"pod_01": {"temp_label": "high", "co2_label": "normal"}, ...}
    sensor_labels: dict = field(default_factory=dict)
    # Alerts appended by each layer
    alerts: list = field(default_factory=list)
    # Written by L3
    anomaly_score: Optional[float] = None
    anomaly_scores_by_model: dict = field(default_factory=dict)

    def is_pod_trustworthy(self, pod_id: str, allow_stuck_suspect: bool = False) -> bool:
        status = self.data_quality.get(pod_id)
        if status == "ok":
            return True
        if allow_stuck_suspect and status == "stuck_suspect":
            return True
        return False

    def add_alert(self, layer, rule_id, severity, message, pod_id=None):
        self.alerts.append(Alert(layer, rule_id, severity, message, pod_id))
