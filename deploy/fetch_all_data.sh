#!/usr/bin/env bash
# Pulls the full historical timeseries (from the beginning of time to now) for
# all 3 pod devices from ThingsBoard via REST API, saving each as its own JSON
# file. Includes both the raw sensor fields and their matching hardware-alarm
# _active fields, so the data can be correctly gated the same way
# converters.py does when turning it into training data.
#
# Usage:
#   export TB_USERNAME=...
#   export TB_PASSWORD=...
#   ./fetch_all_data.sh
#
# Device names / host below match config.yaml's thingsboard section as of
# writing -- double check they still match before running.
set -euo pipefail

HOST="http://localhost:8080"
NOW_MS=$(date +%s%3N)
START_MS=0   # from the very beginning = "all data"

if [ -z "${TB_USERNAME:-}" ] || [ -z "${TB_PASSWORD:-}" ]; then
  echo "Set TB_USERNAME and TB_PASSWORD environment variables first." >&2
  exit 1
fi

TOKEN=$(curl -s -X POST "$HOST/api/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$TB_USERNAME\",\"password\":\"$TB_PASSWORD\"}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

get_device_id() {
  curl -s -G "$HOST/api/tenant/devices" \
    -H "X-Authorization: Bearer $TOKEN" \
    --data-urlencode "deviceName=$1" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['id']['id'])"
}

fetch_device() {
  local device_name=$1 keys=$2 outfile=$3
  local device_id
  device_id=$(get_device_id "$device_name")
  echo "device $device_name -> $device_id"
  curl -s -G "$HOST/api/plugins/telemetry/DEVICE/$device_id/values/timeseries" \
    -H "X-Authorization: Bearer $TOKEN" \
    --data-urlencode "keys=$keys" \
    --data-urlencode "startTs=$START_MS" \
    --data-urlencode "endTs=$NOW_MS" \
    --data-urlencode "limit=100000" \
    -o "$outfile"
  echo "saved -> $outfile"
}

# pod_01 environment
fetch_device "POD_67C3" \
  "sht41_temperature,sht41_humidity,scd41_co2,sgp40_voc,alarm_temperature_c_numeric_high_threshold_active,alarm_relative_humidity_percent_numeric_high_threshold_active,alarm_co2_ppm_numeric_high_threshold_active,alarm_voc_index_numeric_high_threshold_active" \
  pod1_full.json

# pod_02 activity
fetch_device "POD_3C60" \
  "bh1750_illuminance,pir_motion,reed_contact,alarm_illuminance_lux_numeric_high_threshold_active,alarm_contact_state_active_value_active" \
  pod2_full.json

# pod_03 equipment
fetch_device "POD_ABD1" \
  "ds18b20_temperature,ina219_current,adxl345_vibration,alarm_temperature_c_numeric_high_threshold_active,alarm_current_ma_numeric_high_threshold_active,alarm_vibration_rms_g_numeric_high_threshold_active" \
  pod3_full.json
