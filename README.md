# IMU Air Writing Demo

Wrist IMU air writing. Orientation aims a short virtual pen, a page is locked
in space, and the stroke is projected onto that plane. A small EMNIST letter
model reads the 28x28 raster.

Accel is not integrated. This is not inertial dead-reckoning.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python page_projection.py
```

| Key | Action |
|-----|--------|
| R | canned loop |
| T | live IMU (needs the band advertising) |
| D | dark / light |
| arrows | pan |
| drag | orbit |
| wheel | zoom |
| space | pause |
| Q / Esc | quit |

Live mode scans for a band named `BAND` (or the SY manufacturer marker) over BLE.

## Files

| File | Role |
|------|------|
| `page_projection.py` | demo window |
| `motion.py` | hinge tip to locked page |
| `segmenter.py` | trim the lead-in of a stroke |
| `rasterize.py` | stroke to 28x28 |
| `classifier.py` | EMNIST A-Z CNN |
| `ble_client.py` | IMU samples over BLE |
| `models/emnist_letters_cnn.pt` | weights |
