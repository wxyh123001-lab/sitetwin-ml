"""
Standalone smoke test: exercises the REAL _push_alarms_to_tb() code path
(not a reimplementation) with two hand-built alerts -- one at "critical"
severity (should push a TB alarm AND fire test_output on the target pod),
one at "warning" (should push a TB alarm but skip test_output entirely) --
to confirm the severity gate actually works, not just that send_rpc() by
itself can reach a Pod.

Uses _tb_setup() from main.py to get real login + device resolution, same
as the full pipeline would.
"""
import os
import yaml
from main import _tb_setup, _push_alarms_to_tb

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

username = os.environ.get("TB_USERNAME")
password = os.environ.get("TB_PASSWORD")
if not username or not password:
    raise SystemExit("Set TB_USERNAME and TB_PASSWORD environment variables first.")

client, poll_interval, devices, ml_advisor_device_id = _tb_setup(config)
severity_map = config["thingsboard"]["severity_map"]

fake_result = {
    "alert_state": "active",
    "alerts": [
        {
            "rule_id": "manual_test_critical",
            "severity": "critical",
            "message": "Manual smoke test -- critical, should trigger test_output.",
            "layer": "test",
            "pod_id": None,
            "pod_ids": ["pod_02"],  # POD_6647
        },
        {
            "rule_id": "manual_test_warning",
            "severity": "warning",
            "message": "Manual smoke test -- warning, should NOT trigger test_output.",
            "layer": "test",
            "pod_id": None,
            "pod_ids": ["pod_02"],
        },
    ],
}

print("Pushing one critical + one warning test alert...")
print("Watch POD_6647 -- LED/buzzer should pulse ~5s ONCE (for the critical "
      "alert only), and check the console below for two 'test_output ... "
      "skipped' or success lines matching that expectation.")
_push_alarms_to_tb(client, devices, ml_advisor_device_id, severity_map, fake_result)
print("Done. Check ML_ADVISOR's Alarms list in TB for both rule_ids.")