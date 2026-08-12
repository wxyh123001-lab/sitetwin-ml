"""
Fusion layer.
Responsibilities:
  1. Merge all alerts produced by every layer this round
  2. Severity is decided by L1/L2 only; an L3-only trigger is always lowest
     priority (info), the anomaly score is never mapped directly to a severity
  3. If the L3 score coincides with an L1/L2 alert, that alert's severity can
     be bumped up one level (confidence boost, L3 still doesn't decide severity)
  4. Emit the unified ml_output format
"""


class AlertFusion:
    def __init__(self, config, active_alerts_memory):
        self.cfg = config["fusion"]
        self.rank = self.cfg["severity_rank"]  # ["info","warning","critical"]
        # active_alerts_memory: {(pod_id, rule_id): last_seen_ts}
        # for suppressing repeat pushes of an ongoing event; only a structural stub in this simplified implementation
        self.active_alerts_memory = active_alerts_memory

    def merge(self, snapshot):
        alerts = snapshot.alerts

        if self.cfg["escalate_on_l3_corroboration"] and snapshot.anomaly_score is not None:
            l3_high = snapshot.anomaly_score >= 0.5
            if l3_high:
                for a in alerts:
                    if a.layer in ("L1", "L2") and a.severity != "critical":
                        a.severity = self._escalate(a.severity)
                        a.message += " (L3 independently flagged this combination as rare, confidence boosted)"

        top_severity = "info"
        for a in alerts:
            if self.rank.index(a.severity) > self.rank.index(top_severity):
                top_severity = a.severity

        triggered_rules = [f"{a.layer}:{a.rule_id}" for a in alerts]

        ml_output = {
            "timestamp": snapshot.timestamp.isoformat(),
            "alert_state": top_severity if alerts else "normal",
            "anomaly_score": snapshot.anomaly_score,
            "anomaly_scores_by_model": snapshot.anomaly_scores_by_model,
            "triggered_rules": triggered_rules,
            "alerts": [
                {"layer": a.layer, "rule_id": a.rule_id, "severity": a.severity,
                 "message": a.message, "pod_id": a.pod_id}
                for a in alerts
            ],
        }
        return ml_output

    def _escalate(self, severity):
        idx = self.rank.index(severity)
        return self.rank[min(idx + 1, len(self.rank) - 1)]
