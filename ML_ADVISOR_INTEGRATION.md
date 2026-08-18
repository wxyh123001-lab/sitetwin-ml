# ML_ADVISOR ThingsBoard Integration — Change Notes

Branch: `feature/ml-advisor-tb-integration` (based on `master` @ `bdf82bc`)
Author: Yicheng
Purpose: actually wire L2/L3 output into ThingsBoard, replacing a
`create_alarm` call path that had never been verified against a live
instance and whose target didn't match the agreed design.

## Background: what was wrong before this change

The existing `create_alarm` path had two separate problems:

1. **Never verified against a live ThingsBoard instance.** The original
   docstring in `thingsboard_client.py` said outright: "NOT covered by
   thingsboard_api_reference.md ... has not been verified against a live
   instance yet."
2. **Wrong originator.** The old code split each alert across
   `result["alerts"][i]["pod_ids"]` and attached one alarm per pod directly
   to **that Pod's own device** (`main.py`'s original `_push_alarms_to_tb`:
   `client.create_alarm(dev["device_id"], ...)`), instead of a standalone
   `ML_ADVISOR` identity. That mixes ML inferences with the
   hardware-verified Pod/Coordinator alarms mirrored via the TB Rule Chain
   on the same device entity -- there's no way to tell "a sensor actually
   crossed a threshold" from "the AI inferred this from combined signals."
   It also splits a single cross-Pod L2 scenario (e.g.
   `equip_overheat_air_quality`) into multiple unrelated alarm records on
   different devices, losing the "this was a cross-device correlation"
   context entirely.

`config.yaml` also never had an `ML_ADVISOR` device configured at all --
directly explains why "the code calls the API but nothing shows up in TB."

## What changed

### 1. New standalone `ML_ADVISOR` device identity

Created a device named `ML_ADVISOR` in ThingsBoard by hand (not tied to any
real hardware), UUID: `3662fc90-9b28-11f1-8d0f-dd659b48fe4b`. Every
ML-originated alarm now attaches here, physically separate from the
Pod/Coordinator hardware alarm system.

### 2. `config.yaml`

Added `thingsboard.ml_advisor_device_id` holding that UUID. Not a secret
(same category as the existing `device_to_pod` mapping -- just an
identifier), safe to commit as-is.

### 3. `thingsboard_client.py`

Added `push_telemetry_by_token()` -- pushes a heartbeat to
`/api/v1/{token}/telemetry` using the device's own Access Token (not the
tenant JWT used everywhere else). Purely to keep `ML_ADVISOR` showing
Active in the TB UI (see "Heartbeat" below); completely independent from
the admin-level `create_alarm` Alarm API.

### 4. `main.py`

- `_tb_setup()`: now also reads and returns `ml_advisor_device_id`; exits
  with a clear error if it's missing, rather than silently falling back to
  something else.
- `_push_alarms_to_tb()`: no longer loops over `pod_ids` attaching one
  alarm per Pod device. Now attaches a single alarm to
  `ml_advisor_device_id`, with the contributing Pods listed in
  `details.contributing_pods` -- preserving the "this was a cross-Pod
  judgment" context instead of discarding it.
- New `_send_ml_advisor_heartbeat()` -- see "Heartbeat" below.
- Both call sites (cold-start phase `_cold_start_thingsboard` and the
  steady-state loop `run_with_thingsboard_data`) updated to pass the new
  parameter through and call the heartbeat once per poll.

### 5. Heartbeat (keeps `ML_ADVISOR` shown as Active)

`create_alarm` goes through the tenant-JWT-authenticated admin Alarm API --
"create this record on the device's behalf" -- which does *not* make the
device itself show Active. TB's Active/Inactive status only reflects
whether the device *itself* has pushed data over a device-level protocol.
To make `ML_ADVISOR`'s Active state a meaningful signal for "is the ML
service actually running" (loosely mirroring how `sitetwin-gateway-debug`
stays Active via bridge.py's persistent MQTT connection -- though this is
REST polling, not a persistent connection, so there's a delay bounded by
the device profile's inactivity timeout rather than being instant), added
`_send_ml_advisor_heartbeat()`, called once per poll in both loops.

**Needs a new environment variable**: `TB_ML_ADVISOR_TOKEN`, set to
`ML_ADVISOR`'s own Device Access Token (not the tenant username/password).
This is purely cosmetic -- if unset, it just prints one warning and
continues; alarm pushing is unaffected, so this is not a hard `SystemExit`.

### 6. New `test_alarm_push.py`

Standalone smoke test that bypasses the full pipeline and only exercises
`create_alarm` itself against a live instance, mounted on `ML_ADVISOR`.
Used to isolate "is this an auth problem, a permissions problem, or a body
format problem" without needing L2/L3 to actually trigger anything. Not
part of the normal run flow -- purely a debugging tool; whether to keep it
long-term is open for discussion.

## Verification status

- ✅ `test_alarm_push.py` ran clean end-to-end: login succeeded,
  `create_alarm` returned a full TB Alarm object, `originator.id` confirmed
  as `ML_ADVISOR`, and the alarm showed up correctly in both the TB
  notification and the Alarms list.
- ✅ Heartbeat verified: with `TB_ML_ADVISOR_TOKEN` set, `ML_ADVISOR`'s
  status flipped from Inactive to Active.
- ✅ `python3 main.py --source thingsboard` ran cleanly into the main
  polling loop: all three real Pod devices resolved correctly, and the
  L2+L3 pipeline loaded successfully (after upgrading `scikit-learn` to
  1.7.2 -- see "Incidental findings" below for why that was needed).
- ⚠️ **Not yet verified**: the full automatic path -- L2/L3 actually
  triggering during the real polling loop, then automatically calling
  `_push_alarms_to_tb()`, then landing in TB. No real Pod hardware was
  online producing data during this session, and no trigger condition was
  manually constructed, so this step is only verified *by equivalence*
  (both paths ultimately call the same `create_alarm`), not by an actual
  live run of the automatic path itself.

## Incidental findings (out of scope for this change, noted for the record)

- **Model file version mismatch**: `models/lof.joblib` was trained and
  saved under scikit-learn 1.7.2, but the ZBook environment had 1.2.2
  installed, causing
  `AttributeError: Can't get attribute 'EuclideanDistance64'` on load.
  Worked around locally with `pip install --upgrade scikit-learn==1.7.2`,
  but this is a side effect of the `926c16f` temporary model-commit change
  -- nobody has aligned environment versions across machines yet. The
  README already flags the model commit as temporary; worth adding a note
  about the version-pinning risk too.
- `layers/l2_context.py`'s `vibration_mechanical_fault` scenario had its
  trigger condition loosened in `8123773` (dropped the
  `current_label == "normal"` requirement). This could now double-fire
  alongside `equip_stall_risk` / `equip_high_load` when vibration is
  abnormal *and* current is also elevated at the same time. Not sure if
  this was intentional -- worth confirming with Yang.

## Open questions

1. Should `test_alarm_push.py` stay in the repo as a debugging tool, or
   get removed before merging?
2. Is the `ML_ADVISOR` heartbeat feature actually worth keeping? It's
   purely cosmetic (makes the TB UI look more sensible) and doesn't affect
   any functional behavior -- fine to revert entirely if not wanted.
3. The automatic trigger path (L2/L3 -> auto-push) still hasn't been
   exercised for real. Worth testing once with either real hardware or a
   manually constructed edge case, to confirm `_push_alarms_to_tb` also
   works correctly when called from the automatic path, not just the
   manual smoke-test script.