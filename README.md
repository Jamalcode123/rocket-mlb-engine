[README.md](https://github.com/user-attachments/files/28814440/README.md)
# Rocket MLB — Baseball Analytics & Simulation Engine

A Python engine that ingests live Major League Baseball data and runs large-scale
Monte Carlo simulations to produce probabilistic projections for player and team
outcomes (pitcher strikeouts, hitter total bases, team run totals, and more).

Built independently as a self-directed project to learn data engineering,
external API integration, and statistical simulation in Python.

---

## What it does

The system is split into two cleanly separated layers:

**1. Data Layer (`rocket_data_layer.py`)**
An automated pipeline that:
- Pulls schedules, rosters, lineups, and box scores from the **MLB Stats API**
- Pulls pitch-level and advanced metrics via **Statcast / pybaseball**
- Pulls hourly forecasts from the **Open-Meteo** weather API
- Normalizes everything into typed data structures (`PitcherProfile`,
  `HitterProfile`, `BullpenProfile`, `TeamOffenseProfile`, `GameEnvironment`)
- Classifies players into roles with a dedicated `RoleEngine`
- Caches API responses to disk to avoid redundant network calls

**2. Simulation Engine (`rocket_sim_engine.py`)**
A Monte Carlo framework (150,000 iterations per matchup) with dedicated
simulators for:
- Starting pitcher outs / innings pitched and strikeouts
- Hitter total bases, hits, and runs + RBI
- Stolen bases
- Team run totals (first 5 innings and full game)

It consumes the `GameEnvironment` objects from the data layer and outputs
probability distributions and projections for a full day's slate of games.

---

## Tech stack

- **Language:** Python 3
- **Libraries:** `requests`, `pandas`, `numpy`, `pybaseball`
- **Data sources (all free / no auth):** MLB Stats API, Open-Meteo, Baseball Savant / Statcast
- **Concepts:** object-oriented design, typed dataclasses, REST API integration,
  response caching, Monte Carlo simulation

---

## Example usage

```python
from rocket_data_layer import RocketDataLayer
from rocket_sim_engine import RocketSimEngine

# 1. Build the day's slate from live data
rdl = RocketDataLayer()
slate = rdl.run_daily_pipeline("2026-06-15")

# 2. Run the Monte Carlo simulations
sim = RocketSimEngine()
results = sim.run_full_slate(slate)
```

---

## Notes

This is an independent learning project I designed and built myself, end to end.
It reflects hands-on experience with Python, working with multiple live data APIs,
and building statistical models from scratch.
