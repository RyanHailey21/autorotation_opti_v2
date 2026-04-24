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

**Two-stage process:** coarse screening of all airfoils → refine top 3 with tighter tolerances. The refinement stage regularly produces worse results than coarse due to gradient landscape differences between the two BEM fidelity levels — this is expected. `best_score` is seeded from the best coarse result so the overall best is preserved regardless of stage.

**Three simulation fidelity levels** used within `optimize()`:
- `cfg_coarse`: n_span=5, induction_max_iter=4, dt=0.03 — used inside the CasADi optimizer graph
- `cfg_refine`: n_span=7, induction_max_iter=6, dt=0.02 — used for refinement stage optimizer graph
- `cfg_eval`: n_span=20, induction_max_iter=35, dt=0.01 — used for all final `simulate_drop` evaluations

**Physics pipeline** (called from both optimizer and simulator):
1. `polars.py` — generates or loads `Cl/Cd` tables vs. Re and α (NeuralFoil or XFoil)
2. `rotor.py` — Blade-Element Momentum (BEM) theory with iterative induction, Prandtl tip/hub loss corrections
3. `aero.py` — Reynolds-aware fallback `Cl/Cd` model with Viterna post-stall blending; **only active when no `polar_npz_path` is set**
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

## Optimizer Behavior Notes

- **IPOPT warm-start**: default guess uses `t_final=7.0` (not free-fall ~2s), `v_final=3.0 m/s`, `omega_final=120 rad/s`. Coarse results are chained as warm starts for the next airfoil, but only on `Solve_Succeeded` or `Solved_To_Acceptable_Level`. If a chained warm start produces a failed solve, the optimizer retries from the default warm start.
- **Prandtl tip loss**: CasADi expression clamps `exp(-f)` below `1 - 1e-6` before passing to `acos` to avoid infinite gradients at the tip station.
- **Refinement degradation**: the printed warning "refine degraded vs coarse" is expected for some airfoils and is handled correctly — the coarse result is kept as a candidate and `best_score` comparison uses `cfg_eval` for both.
- **Material density** in `RotorDesign` defaults to 650 kg/m³, representing a hollow-shell 3D printed structure. Solid PLA is ~1250 kg/m³; adjust if using high-infill prints.
- **Low-Re accuracy**: blade sections operate at Re ~15,000–50,000. NeuralFoil is trained on XFoil data but smooths over laminar separation bubbles common at these Re. Absolute performance predictions are optimistic; relative airfoil rankings are more reliable.

## Dependencies

Core: `numpy >= 1.24`, `scipy >= 1.10`, `aerosandbox >= 4.2` (provides NeuralFoil, `asb.Opti`, and optional XFoil integration). `matplotlib` is used by `reporting.py` but failures are handled gracefully. XFoil backend requires the `xfoil` executable on PATH.
