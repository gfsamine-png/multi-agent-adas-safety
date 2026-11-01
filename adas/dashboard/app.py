from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from adas.dashboard.summary import summarize  # noqa: E402
from adas.storage.incident_db import IncidentDb  # noqa: E402

st.set_page_config(page_title="ADAS Incident Dashboard", layout="wide")
st.title("ADAS incident dashboard")
st.caption("Alerts only. The system never controls the vehicle.")

db_path = st.sidebar.text_input("Database file", "outputs/incidents.db")
if not Path(db_path).exists():
    st.info(
        "No database yet. Create one with:\n\n"
        "`py -m adas.storage.store_alerts outputs\\observations.jsonl`"
    )
    st.stop()

db = IncidentDb(db_path)
runs = db.runs()
if not runs:
    st.info("The database has no runs yet.")
    st.stop()

labels = {
    r["id"]: f"Run {r['id']} | {r['source']} | {r['created_at']}"
    + (" | SIMULATED" if r["simulated"] else "")
    for r in runs
}
run_id = st.sidebar.selectbox("Run", list(labels), format_func=labels.get)
run = next(r for r in runs if r["id"] == run_id)
alerts = db.alerts(run_id)

if run["simulated"]:
    st.warning(
        "SIMULATED run: the lane offset contains a synthetic drift. "
        "Its lane-departure alerts are a test of the logic, not road results."
    )
st.caption(f"Source: {run['source']}  |  {run['note']}")

s = summarize(alerts)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total alerts", s["total"])
c2.metric("Collision alerts", s["collisions"])
c3.metric("Lane departure alerts", s["lane_departures"])
c4.metric("Worst collision level", s["worst_collision_level"])

if not alerts:
    st.success("No alerts in this run.")
    st.stop()

df = pd.DataFrame(alerts)

st.subheader("Timeline")
chart = (
    alt.Chart(df)
    .mark_circle(size=250)
    .encode(
        x=alt.X("timestamp_s:Q", title="Time in video (s)"),
        y=alt.Y("kind:N", title=None),
        color=alt.Color("level:N", title="Level"),
        tooltip=["timestamp_s", "kind", "level", "track_id", "ttc_s", "reason"],
    )
    .properties(height=180)
)
st.altair_chart(chart, use_container_width=True)

st.subheader("Alert log")
st.dataframe(
    df[["timestamp_s", "kind", "level", "track_id", "ttc_s", "offset", "reason", "can_data"]],
    use_container_width=True,
    hide_index=True,
)