#!/usr/bin/env bash
# Installs SiteTwin ML as a systemd service on the Pi: creates a venv, installs
# runtime dependencies, sets up a credentials env file (if missing), and
# writes + enables the systemd unit so it starts on boot and restarts on
# crash. Safe to re-run (won't overwrite an existing env file).
#
# Usage: sudo ./deploy/install_service.sh
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
  echo "Run with sudo: sudo $0" >&2
  exit 1
fi

# Project dir = this script's parent (deploy/..), so it works no matter where
# the repo was cloned to.
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_USER="${SUDO_USER:-pi}"
ENV_FILE="/etc/sitetwin-ml.env"
UNIT_FILE="/etc/systemd/system/sitetwin-ml.service"
VENV_DIR="$INSTALL_DIR/.venv"

echo "Installing SiteTwin ML service"
echo "  project dir : $INSTALL_DIR"
echo "  run as user : $SERVICE_USER"

# 1. Isolated venv + runtime deps only (not requirements.txt's optuna/
#    matplotlib -- those are for the offline model-selection experiments,
#    not for running main.py). A venv is used rather than the system
#    interpreter because recent Raspberry Pi OS (Bookworm+) blocks `pip
#    install` into the system environment by default (PEP 668).
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -q pyyaml scikit-learn numpy joblib requests
chown -R "$SERVICE_USER":"$SERVICE_USER" "$VENV_DIR"

# 2. Credentials file. Only created if missing -- never overwrites real
#    credentials on a re-run. Mode 600: only root can read it; systemd reads
#    it as root before dropping to $SERVICE_USER.
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
# ThingsBoard credentials for SiteTwin ML. Fill these in, then:
#   sudo systemctl restart sitetwin-ml
TB_USERNAME=
TB_PASSWORD=
EOF
  chmod 600 "$ENV_FILE"
  echo "Created $ENV_FILE -- edit it and fill in TB_USERNAME / TB_PASSWORD."
else
  echo "$ENV_FILE already exists, leaving it untouched."
fi

# 3. systemd unit
cat > "$UNIT_FILE" <<EOF
[Unit]
Description=SiteTwin ML (L2-L3 anomaly detection, ThingsBoard polling)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
WorkingDirectory=$INSTALL_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_DIR/bin/python3 $INSTALL_DIR/main.py --source thingsboard
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 4. Enable on boot, start now if credentials are already filled in
systemctl daemon-reload
systemctl enable sitetwin-ml.service

if grep -qE '^(TB_USERNAME|TB_PASSWORD)=$' "$ENV_FILE"; then
  echo ""
  echo "TB_USERNAME/TB_PASSWORD are still empty in $ENV_FILE."
  echo "Fill them in (and check config.yaml's thingsboard.host/device_to_pod), then run:"
  echo "  sudo systemctl start sitetwin-ml"
else
  systemctl restart sitetwin-ml.service
  echo ""
  echo "Service started. Useful commands:"
  echo "  systemctl status sitetwin-ml"
  echo "  journalctl -u sitetwin-ml -f"
fi
