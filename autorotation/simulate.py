import argparse
import numpy as np

try:
    from .models import Environment, Body, RotorDesign, SimConfig, SimResult
    from .rotor import aero_loads, rotor_inertia, rotor_mass
except ImportError:
    # Support direct script execution: python autorotation/simulate.py
    from models import Environment, Body, RotorDesign, SimConfig, SimResult
    from rotor import aero_loads, rotor_inertia, rotor_mass


def simulate_drop(
    design: RotorDesign,
    env: Environment,
    body: Body,
    cfg: SimConfig,
    progress_every_s: float | None = None,
) -> SimResult:
    t = 0.0
    h = cfg.drop_height_m
    v = cfg.v0_down_m_s
    omega = max(cfg.omega0_rad_s, 0.0)

    i_rotor = max(rotor_inertia(design), 1e-7)
    total_mass_kg = body.mass_kg + rotor_mass(design)

    ts, hs, vs, rpms = [], [], [], []
    thrusts, torques = [], []

    next_progress_t = progress_every_s if progress_every_s and progress_every_s > 0 else None
    while t < cfg.t_max_s and h > 0.0:
        loads = aero_loads(design=design, env=env, cfg=cfg, omega=omega, v_down=v)

        # Keep solution in low-Mach regime where this model is valid.
        a_sound = 343.0
        tip_mach = omega * design.radius_m / a_sound
        if tip_mach > cfg.max_tip_mach:
            omega *= cfg.max_tip_mach / max(tip_mach, 1e-9)

        drag_body = 0.5 * env.rho * body.cd_body * body.area_body_m2 * v * abs(v)
        a_down = env.g - (loads.thrust_up_n + drag_body) / total_mass_kg
        v = max(v + a_down * cfg.dt_s, 0.0)
        h = h - v * cfg.dt_s

        domega = (loads.torque_nm - design.friction_coef * omega * abs(omega)) / i_rotor
        omega = max(omega + domega * cfg.dt_s, 0.0)

        t += cfg.dt_s
        ts.append(t)
        hs.append(max(h, 0.0))
        vs.append(v)
        rpms.append(omega * 60.0 / (2.0 * np.pi))
        thrusts.append(loads.thrust_up_n)
        torques.append(loads.torque_nm)

        if v <= 1e-6 and loads.thrust_up_n > total_mass_kg * env.g:
            # Rare nonphysical hover in this simplified model; keep simulation stable.
            v = 1e-3

        if next_progress_t is not None and t >= next_progress_t:
            print(f"[simulate] t={t:.1f}s h={max(h,0.0):.2f}m v={v:.2f}m/s rpm={rpms[-1]:.0f}")
            next_progress_t += progress_every_s

    ts = np.asarray(ts)
    hs = np.asarray(hs)
    vs = np.asarray(vs)
    rpms = np.asarray(rpms)
    thrusts = np.asarray(thrusts)
    torques = np.asarray(torques)

    max_rpm = float(np.max(rpms)) if rpms.size else 0.0
    if rpms.size:
        i0 = int(0.8 * len(rpms))
        mean_rpm_last = float(np.mean(rpms[i0:]))
    else:
        mean_rpm_last = 0.0

    return SimResult(
        fall_time_s=float(t),
        impact_speed_m_s=float(v),
        max_rpm=max_rpm,
        mean_rpm_last_20pct=mean_rpm_last,
        trace_t=ts,
        trace_h=hs,
        trace_v=vs,
        trace_rpm=rpms,
        trace_thrust_n=thrusts,
        trace_torque_nm=torques,
    )


def _default_design():
    return RotorDesign()


def _parse():
    p = argparse.ArgumentParser(description="Run a single passive-autorotation drop simulation.")
    p.add_argument("--radius", type=float, default=0.24)
    p.add_argument("--chord-root", type=float, default=0.06)
    p.add_argument("--chord-tip", type=float, default=0.03)
    p.add_argument("--twist-root", type=float, default=18.0)
    p.add_argument("--twist-tip", type=float, default=4.0)
    p.add_argument("--pitch", type=float, default=8.0)
    p.add_argument("--camber", type=float, default=0.04)
    p.add_argument("--camber-pos", type=float, default=0.4)
    p.add_argument("--thickness", type=float, default=0.11)
    p.add_argument("--polar-npz", type=str, default=None, help="Optional XFoil polar table npz path.")
    p.add_argument("--progress", action="store_true", help="Print lightweight periodic progress.")
    p.add_argument("--progress-interval", type=float, default=0.5, help="Seconds between progress messages.")
    return p.parse_args()


def main():
    args = _parse()

    design = _default_design()
    design.radius_m = args.radius
    design.chord_root_m = args.chord_root
    design.chord_tip_m = args.chord_tip
    design.twist_root_deg = args.twist_root
    design.twist_tip_deg = args.twist_tip
    design.pitch_collective_deg = args.pitch
    design.camber = args.camber
    design.camber_pos = args.camber_pos
    design.thickness = args.thickness

    cfg = SimConfig()
    cfg.polar_npz_path = args.polar_npz
    progress_every_s = args.progress_interval if args.progress else None
    res = simulate_drop(design, Environment(), Body(), cfg, progress_every_s=progress_every_s)
    print(f"fall_time_s={res.fall_time_s:.3f}")
    print(f"impact_speed_m_s={res.impact_speed_m_s:.3f}")
    print(f"max_rpm={res.max_rpm:.1f}")
    print(f"mean_rpm_last_20pct={res.mean_rpm_last_20pct:.1f}")


if __name__ == "__main__":
    main()
