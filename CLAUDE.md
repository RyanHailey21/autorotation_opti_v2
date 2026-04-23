# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Physics-based optimizer for passive autorotating helicopter rotors used on dropped drones. The goal is to maximize fall time (time-to-ground) from 18.3 m by designing a rotor that spins up naturally from falling airflow. Conceptual design tool — fast and practical, not high-fidelity.

## Running the Code

```bash
# Full optimization run (recommended entry point)
python -m autorotation.optimize --backend neuralfoil --model-size medium --iters 3

# Evaluate a single hardcoded design
python -m autorotation.simulate

# Generate airfoil polars manually
python -m autorotation.polars --backend neuralfoil --out data/polar_naca.npz --camber 0.04 --camber-pos 0.4 --thickness 0.12

# Skip optimization cache (force recompute)
python -m autorotation.optimize --no-opt-cache
```

All commands use the `.venv` virtual environment. No build step is required.

## Architecture

The optimizer runs in two nested loops:

**Outer loop** (`optimizer.py`) — iterates over a grid of NACA 4-digit airfoils (camber × camber position × thickness combinations).

**Inner loop** (`optimizer.py: optimize_geometry()`) — for each airfoil, runs direct-transcription geometry optimization via AeroSandbox (`asb.Opti` wrapping CasADi/IPOPT) over 6 design variables: `radius_m`, `chord_root_m`, `chord_tip_m`, `twist_root_deg`, `twist_tip_deg`, `pitch_collective_deg`.

**Two-stage process:** coarse screening of all airfoils → refine top 3 with tighter tolerances.

**Physics pipeline** (called from both optimizer and simulator):
1. `polars.py` — generates or loads `Cl/Cd` tables vs. Re and α (NeuralFoil or XFoil)
2. `rotor.py` — Blade-Element Momentum (BEM) theory with iterative induction, Prandtl tip/hub loss corrections
3. `aero.py` — Reynolds-aware fallback `Cl/Cd` model with Viterna post-stall blending
4. `simulate.py` — time-marching drop dynamics that couples rotor spin-up to body fall

**Reporting** (`reporting.py`) — writes CSV summary and matplotlib plots to `outputs/reports/`.

## Caching

- `.cache/polars/` — one `.npz` per airfoil + backend + Re/α grid; reused across runs
- `.cache/optimize/` — warmstart states and final geometries; keyed by config hash; disable with `--no-opt-cache`
- `data/polar_naca.npz` — pre-computed example polar; pass via `SimConfig.polar_npz_path` to skip recomputation

## Key Data Structures (`models.py`)

`Environment`, `Body`, `RotorDesign`, `SimConfig`, `SimResult` are the dataclasses that flow through the system. `RotorDesign` carries the 6 geometry variables. `SimConfig` carries solver settings, polar path, and backend choice.

## Optimization Objective

Maximize fall time (s), subject to:
- Solidity: `0.04 ≤ σ ≤ 0.35`
- RPM penalty: `+0.004 × (RPM − 1800)` if RPM > 1800
- Impact speed penalty: `+0.12 × (v − 6.0)` if v > 6 m/s

## Dependencies

Core: `numpy >= 1.24`, `scipy >= 1.10`, `aerosandbox >= 4.2` (provides NeuralFoil, `asb.Opti`, and optional XFoil integration). `matplotlib` is used by `reporting.py` but failures are handled gracefully. XFoil backend requires the `xfoil` executable on PATH.
