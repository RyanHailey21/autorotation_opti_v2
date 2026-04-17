# Passive Autorotation Optimizer

This project optimizes a passive autorotating rotor for a small dropped drone.

The simulator models:
- Time-domain drop from a specified height.
- Rotor spin-up from near-zero RPM due to upward inflow during descent.
- Spanwise BEM aerodynamics with:
  - iterative axial/tangential induction factors,
  - Prandtl tip/hub loss factors,
  - Reynolds-varying section coefficients,
  - post-stall drag/lift blending.
- Rotor torque and axial force coupling back into body descent dynamics.

The optimizer searches rotor geometry and section-shape parameters to maximize total fall time.

## Model Scope

This is a fast conceptual design tool, not a full CFD/FSI solver.

Included physics:
- Blade-element forces resolved per span station
- Reynolds-dependent lift/drag model
- Rotor inertial spin-up dynamics
- Body parasitic drag

Not included (yet):
- Dynamic stall/hysteresis
- Unsteady wake or free-vortex wake
- 6-DOF attitude coupling

## Quick Start

```bash
python -m autorotation.optimize --backend neuralfoil --model-size medium --iters 3
```

This runs an outer loop over airfoils and a direct-transcription inner geometry optimization for each airfoil.

To inspect a single design:

```bash
python -m autorotation.simulate
```

## Airfoil Loop

The optimizer now:
- Iterates over a grid of NACA-style airfoils
- Generates or reuses cached polars for each airfoil
- Runs a coarse direct-transcription geometry optimization for each airfoil
- Refines the top airfoils in a second direct-transcription pass
- Reports the best combined airfoil + rotor geometry
- Caches optimization-stage results so repeated runs can skip already-solved airfoils

Code organization:
- [`autorotation/optimize.py`](/c:/Users/ryanh/Desktop/autorotation_opti_v2/autorotation/optimize.py): thin CLI entrypoint
- [`autorotation/optimizer.py`](/c:/Users/ryanh/Desktop/autorotation_opti_v2/autorotation/optimizer.py): optimization engine and caching
- [`autorotation/reporting.py`](/c:/Users/ryanh/Desktop/autorotation_opti_v2/autorotation/reporting.py): CSV summaries and plots

Note:
- The inner `Opti` problem uses the actual drop states (`h`, `v`, `omega`) and BEM-like rotor loads.
- State derivatives are enforced with AeroSandbox derivative constraints rather than manual step equations.
- Warm starts are reused across neighboring airfoils and between coarse/refine stages.
- Variables are explicitly scaled to improve IPOPT behavior.
- Final scoring is still done with the higher-fidelity time-marching simulator.

Example:

```bash
python -m autorotation.optimize --backend neuralfoil --model-size medium --iters 3 --camber-grid 0.00,0.02,0.04 --camber-pos-grid 0.4,0.5 --thickness-grid 0.08,0.12
```

Cached polars are stored under `.cache/polars` by default.
Optimization-stage caches are stored under `.cache/optimize` by default.

To disable optimization caching for a clean rerun:

```bash
python -m autorotation.optimize --backend neuralfoil --model-size medium --iters 3 --no-opt-cache
```

## Optional: Generate One Polar Database Manually

Install optional dependencies:

```bash
pip install -r requirements-xfoil.txt
```

Generate a single polar database with NeuralFoil:

```bash
python -m autorotation.polars --backend neuralfoil --out data/polar_naca.npz --camber 0.04 --camber-pos 0.4 --thickness 0.12
```

Or with XFoil:

```bash
python -m autorotation.polars --backend xfoil --out data/polar_naca.npz --camber 0.04 --camber-pos 0.4 --thickness 0.12
```

## Key Design Variables

- `radius_m`
- `chord_root_m`, `chord_tip_m`
- `twist_root_deg`, `twist_tip_deg`
- `pitch_collective_deg`
- `camber`
- `camber_pos`
- `thickness`

## Optimization Objective

Maximize:
- `fall_time_s`: time from release until ground impact from 18.3 m.

Subject to:
- Practical variable bounds in the optimizer.
