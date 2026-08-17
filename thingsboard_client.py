"""
ThingsBoard REST API client -- raw HTTP calls following thingsboard_api_reference.md.
Response *parsing* (string->number, sensor-key->field, ms->datetime) lives in
converters.py; this file only does the requests themselves.

Credentials are NOT hardcoded (per the doc's security note): username/password
come from the TB_USERNAME / TB_PASSWORD environment variables. Host and the
device->pod mapping are non-secret and live in config.yaml.

NOTE: written against the documented/verified API shapes, but not yet tested
against a live ThingsBoard instance (the Pi is not reachable yet). Once real
access is available, only credentials/host need filling in -- the call shapes
should not need to change.
"""
import requests


class ThingsBoardClient:
    def __init__(self, host, timeout=10):
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.token = None
        self._username = None
        self._password = None

    # ---------- 2. Authentication ----------

    def login(self, username, password):
        """POST /api/auth/login -> JWT token (kept for subsequent calls)."""
        self._username, self._password = username, password
        resp = requests.post(f"{self.host}/api/auth/login",
                             json={"username": username, "password": password},
                             timeout=self.timeout)
        resp.raise_for_status()
        self.token = resp.json()["token"]
        return self.token

    def _headers(self):
        return {"X-Authorization": f"Bearer {self.token}"}

    def _get(self, path, params=None):
        """GET with the auth header; transparently re-login once on 401
        (token expires after ~2.5h, per the doc)."""
        url = f"{self.host}{path}"
        resp = requests.get(url, params=params, headers=self._headers(), timeout=self.timeout)
        if resp.status_code == 401 and self._username:
            self.login(self._username, self._password)
            resp = requests.get(url, params=params, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path, json_body=None):
        """POST with the auth header; same 401 re-login retry as _get."""
        url = f"{self.host}{path}"
        resp = requests.post(url, json=json_body, headers=self._headers(), timeout=self.timeout)
        if resp.status_code == 401 and self._username:
            self.login(self._username, self._password)
            resp = requests.post(url, json=json_body, headers=self._headers(), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ---------- 3. Finding a device's internal ID ----------

    def get_device_id(self, device_name):
        """GET /api/tenant/devices?deviceName=... -> internal UUID string."""
        data = self._get("/api/tenant/devices", params={"deviceName": device_name})
        return data["id"]["id"]

    def list_devices(self, page_size=100, page=0):
        """GET /api/tenant/devices?pageSize=..&page=.. -> full device list page."""
        return self._get("/api/tenant/devices", params={"pageSize": page_size, "page": page})

    # ---------- 5. Querying attributes (static values) ----------

    def get_attributes(self, device_id):
        """GET .../values/attributes/CLIENT_SCOPE -> list of {key, value, ...}."""
        return self._get(
            f"/api/plugins/telemetry/DEVICE/{device_id}/values/attributes/CLIENT_SCOPE")

    # ---------- 4. Querying historical telemetry ----------

    def get_timeseries(self, device_id, keys, start_ts, end_ts, limit=None, agg=None, interval=None):
        """
        GET .../values/timeseries -> {key: [{ts, value}, ...], ...}.
        keys: list of key names (e.g. "sht41_temperature", "scd41_co2"). start_ts/end_ts: epoch ms.
        Optional limit (raise the ~100 default cap) and agg/interval (pre-aggregation).
        """
        params = {"keys": ",".join(keys), "startTs": int(start_ts), "endTs": int(end_ts)}
        if limit is not None:
            params["limit"] = limit
        if agg is not None:
            params["agg"] = agg
        if interval is not None:
            params["interval"] = interval
        return self._get(
            f"/api/plugins/telemetry/DEVICE/{device_id}/values/timeseries", params=params)

    # ---------- 8. Alarms (optional) ----------

    def get_alarms(self, device_id, page_size=100, page=0):
        """GET /api/alarm/DEVICE/{deviceId} -> alarm records for that device."""
        return self._get(f"/api/alarm/DEVICE/{device_id}",
                         params={"pageSize": page_size, "page": page})

    def create_alarm(self, device_id, alarm_type, severity, details=None):
        """
        POST /api/alarm -> create (or update, if one of the same type is already
        ACTIVE for this device) an alarm on a device.

        NOT covered by thingsboard_api_reference.md -- that doc only documents
        reading ThingsBoard's own auto-generated alarms. This follows
        ThingsBoard's general/standard REST API shape and has not been verified
        against a live instance yet.

        severity must be one of ThingsBoard's enum: CRITICAL / MAJOR / MINOR /
        WARNING / INDETERMINATE (see config.yaml's thingsboard.severity_map for
        the info/warning/critical -> this mapping, still a placeholder pending
        confirmation).
        """
        body = {
            "type": alarm_type,
            "originator": {"entityType": "DEVICE", "id": device_id},
            "severity": severity,
            "status": "ACTIVE_UNACK",
        }
        if details:
            body["details"] = details
        return self._post("/api/alarm", body)
