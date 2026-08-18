"""
Standalone smoke test: verify ThingsBoardClient.create_alarm() actually
works against a live instance, without touching sensor data, L2/L3, or the
full main.py pipeline. Bypasses everything except login + one alarm push.
"""
import os
import yaml
from thingsboard_client import ThingsBoardClient

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

tbcfg = config["thingsboard"]
username = os.environ.get("TB_USERNAME")
password = os.environ.get("TB_PASSWORD")
if not username or not password:
    raise SystemExit("Set TB_USERNAME and TB_PASSWORD environment variables first.")

ml_advisor_device_id = tbcfg["ml_advisor_device_id"]

client = ThingsBoardClient(tbcfg["host"])
print(f"logging in to {tbcfg['host']} ...")
client.login(username, password)
print("login OK")

print(f"pushing test alarm to ML_ADVISOR ({ml_advisor_device_id}) ...")
result = client.create_alarm(
    ml_advisor_device_id,
    "test_manual_trigger",
    "WARNING",
    details={
        "message": "Manual smoke test from terminal, no real sensor data involved.",
        "layer": "test",
        "contributing_pods": [],
    },
)
print("create_alarm response:", result)