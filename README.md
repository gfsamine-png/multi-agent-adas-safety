# Multi-Agent AI Automotive Safety System

Educational ADAS prototype combining computer vision, deterministic
safety logic, LangGraph agents and a simulated CAN/ECU layer.

**Not a safety-certified system. It generates alerts only and never
controls a vehicle.**

## Setup (Windows)
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    pytest