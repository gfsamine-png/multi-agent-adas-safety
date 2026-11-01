from __future__ import annotations

import argparse
from pathlib import Path

from adas.pipeline import iter_alert_events
from adas.storage.incident_db import IncidentDb


def main() -> None:
    p = argparse.ArgumentParser(description="Run the alert pipeline and store the alerts.")
    p.add_argument("jsonl", type=Path)
    p.add_argument("--db", type=Path, default=Path("outputs/incidents.db"))
    p.add_argument("--video", default="drive1.mp4")
    p.add_argument("--alert-level", default="moderate")
    p.add_argument("--simulate-drift", action="store_true")
    args = p.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    db = IncidentDb(str(args.db))
    note = f"alert_level={args.alert_level}"
    if args.simulate_drift:
        note += ", SIMULATED lane drift"
    run_id = db.start_run(args.video, note=note, simulated=args.simulate_drift)

    for ev in iter_alert_events(args.jsonl, args.alert_level, args.simulate_drift):
        db.add_alert(
            run_id, ev.timestamp_s, ev.kind, ev.level, ev.reason,
            ev.track_id, ev.ttc_s, ev.offset,
            ev.frame.arbitration_id, ev.frame.data.hex(" "),
        )

    rows = db.alerts(run_id)
    print(f"Run {run_id} ({note}): stored {len(rows)} alert(s) in {args.db}")
    for r in rows:
        print(f"  t={r['timestamp_s']:6.2f}s  {r['kind']:<15} {r['level']:<9} {r['reason']}")
    db.close()


if __name__ == "__main__":
    main()