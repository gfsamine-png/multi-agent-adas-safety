# AI-Based ADAS: Collision and Lane-Departure Warning Prototype

A prototype driver-assistance pipeline that reads **recorded driving video** and raises two kinds of warnings:

- **Forward collision warning (FCW):** a vehicle or pedestrian ahead in our lane that we are approaching too fast.
- **Lane departure warning (LDW):** our own car drifting toward a lane line.

It **only produces alerts. It never controls the vehicle.** It is a portfolio and learning project, not a product, and it has no functional-safety certification.

## How it works

```
video -> YOLOv8n detection -> ByteTrack + ID stitcher -> ego-zone filter
      -> lane detection (OpenCV) -> per-frame observations (JSONL)
      -> time-to-collision (TTC) -> risk level -> alert validator + LDW
      -> simulated CAN frames -> virtual ECU -> SQLite -> Streamlit dashboard
```

| Layer | What it does | Main files |
|---|---|---|
| Detection | YOLOv8n (COCO, pretrained), confidence 0.35, class-agnostic NMS | `adas/perception/detector.py` |
| Tracking | ByteTrack, plus my own `TrackStitcher` that repairs ID flips and removes duplicate boxes | `adas/perception/stitch.py` |
| Ego zone | Drops detections on our own car (e.g. the side mirror detected as "person") | `adas/perception/detector.py` |
| Lanes | Grayscale, blur, Canny, trapezoid mask, Hough lines, line fit; offset as a fraction of lane width | `adas/perception/lanes.py` |
| TTC | Monocular "looming": fit of ln(box width) over 1.5 s gives 1/TTC, with an uncertainty range | `adas/safety/ttc.py` |
| Risk | Rules combining lane position and TTC: unknown / low / moderate / high / critical | `adas/safety/risk.py`, `adas/safety/corridor.py` |
| Validator | Confirm (0.3 s), hold/hysteresis (1.0 s), cooldown (3.0 s) so warnings do not flicker | `adas/safety/validator.py` |
| LDW | Warns at \|offset\| >= 0.30 for 0.5 s | `adas/safety/ldw.py` |
| CAN + ECU | Custom 8-byte frames (counter + XOR checksum), virtual dashboard ECU that validates them | `adas/ecu/` |
| Storage | SQLite tables `runs` and `alerts`, parameterized queries | `adas/storage/` |
| Dashboard | Streamlit page reading the database | `adas/dashboard/` |
| Pipeline | One shared loop used by the replay and storage scripts | `adas/pipeline.py` |
| Tools | Measuring and debugging helpers (ROI viewer, lane debug, alert video renderer) | `tools/` |

### Risk rules (first match wins)

1. TTC has too little data: `unknown`
2. Object is outside our lane: `low`
3. Not approaching: `low`
4. Uncertain: `moderate` if the best-case TTC is 6 s or less, otherwise `low`
5. Approaching: `critical` <= 1.5 s, `high` <= 3.0 s, `moderate` <= 6.0 s, else `low` (capped at `moderate` if the lane is unknown)

The thresholds are experimental values, not taken from a standard.

### CAN message format (custom, not a real vehicle DBC)

| ID | Meaning | Bytes |
|---|---|---|
| `0x300` | Collision alert | counter, level (0-4), track id (2 B), TTC in 0.1 s (2 B, `0xFFFF` = n/a), reserved, checksum |
| `0x301` | Lane departure | counter, side (1 left, 2 right), offset x 1000 (signed 16 bit), reserved, checksum |

The virtual ECU checks the checksum and detects lost or repeated counters.

## Results

Checked on three clips. No ground-truth labels exist, so **no precision or recall is reported**.

| Clip | Description | Lanes found | Alerts |
|---|---|---|---|
| 1 | Road view, 1920x1080, 24 fps, 213 frames | 100 % | 1 real moderate collision alert at 7.29 s (TTC 4.5 s). 44 per-frame level changes became 1 alert. |
| 2 | 4K city street, faint paint, moving horizon | 24-31 % | Not trusted. Documented failure of the classical lane detector. |
| 3 | Vertical 2160x3840, 60 fps, highway (first 960 frames) | 89 % | 0 after the corridor fallback removed one false alarm. |

Other findings:

- The ID stitcher reduced the lead car from 3+ IDs to 2 over 7 s on clip 1.
- The side mirror was detected as "person" in 88 of 213 frames before the ego zone was added.
- YOLO inference took about 22-28 ms per frame (median) on a quiet PC. This is the model only, not the whole pipeline.
- Lane departure is **not seen on any real clip** (the car stays centered). It is unit-tested and demonstrated with `--simulate-drift`, and those runs are stored with `simulated = 1` and shown with a SIMULATED banner in the dashboard.

## Install

Requires Python 3.10+ (developed on Windows with Python 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

The YOLO weights (`yolov8n.pt`) are downloaded by Ultralytics on first run. **Videos are not included in this repository**; put your own clip in `data/videos/`. Check the license of any clip before publishing results.

## Run

```bash
# 1. Detect, track and find lanes; save one JSON line per frame
py -m adas.run_detection --video data\videos\drive1.mp4 --track --lanes --no-display --json-out outputs\observations.jsonl

# 2. Render a video with risk colors and the alert banner
py tools\render_alerts_video.py

# 3. Encode alerts as CAN frames and replay them through the virtual ECU
py -m adas.ecu.replay_can outputs\observations.jsonl

# 4. Store alerts in SQLite (add --simulate-drift to demo LDW with synthetic drift)
py -m adas.storage.store_alerts outputs\observations.jsonl

# 5. Open the dashboard
py -m streamlit run adas\dashboard\app.py
```

Per-camera setup: lane search area and ego zone are command-line options (`--roi-*`, `--split-x`, `--ego-zone`, `--no-ego-zone`, `--max-width`). Check them on a real frame with `tools/show_roi.py` before trusting the lane results.

## Tests

```bash
pytest
```

Unit tests use synthetic data with known answers for every layer (TTC, stitcher, risk, validator, LDW, CAN round trips and corrupted bytes, database, corridor, ego zone, dashboard summary). Replay scripts judge behavior on real data; unit tests only prove the code is correct.

## Limitations

- Tuned and checked on three clips; lane settings are per camera.
- Monocular TTC is noisy and assumes constant closing speed and an object approaching straight at us. Passing vehicles and boxes cut by the frame edge can distort it.
- The lane detector is classical and straight-line: it fails with a moving horizon or faint paint (clip 2).
- LDW threshold (0.30) is not validated on real drift. There is no turn-signal input, so a deliberate lane change would also warn.
- Track IDs are improved, not perfect. Cooldown is per object, so an ID flip could repeat an alert.
- CAN format is my own design; there is no real hardware.
- No functional-safety process; not usable in a vehicle.

## Roadmap

- [ ] Evaluation with labelled events (precision, recall, false alarms per hour) and TTC compared with ground-truth distance
- [ ] A clip with a real close approach and one with real lane drift
- [ ] More robust lanes (horizon estimate, plausibility checks, or a learned model)
- [ ] GitHub Actions CI running `pytest`
- [ ] Dockerfile
