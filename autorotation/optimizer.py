import hashlib
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import numpy as np

try:
    import aerosandbox as asb
    import aerosandbox.numpy as asb_np
except Exception:  # pragma: no cover
    asb = None
    asb_np = None
try:
    import casadi as cs
except Exception:  # pragma: no cover
    cs = None

from .models import Environment, Body, RotorDesign, SimConfig
from .simulate import simulate_drop
from .polars import generate_polar_npz, naca4_from_design
from .reporting import summary_record, write_summary_csv, generate_report_plots

GEOMETRY_BOUNDS = {
    "radius_m": (0.10, 0.40),
    "chord_root_m": (0.020, 0.100),
    "chord_tip_m": (0.010, 0.070),
    "twist_root_deg": (6.0, 35.0),
    "twist_tip_deg": (-4.0, 18.0),
    "pitch_collective_deg": (0.0, 18.0),
}

GEOMETRY_KEYS = list(GEOMETRY_BOUNDS.keys())
_INTERP_COUNTER = 0


def parse_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def default_design() -> RotorDesign:
    return RotorDesign()


def vec_to_design(x: np.ndarray, airfoil: dict[str, float] | None = None) -> RotorDesign:
    d = default_design()
    for i, key in enumerate(GEOMETRY_KEYS):
        setattr(d, key, float(x[i]))

    if airfoil is not None:
        d.camber = float(airfoil["camber"])
        d.camber_pos = float(airfoil["camber_pos"])
        d.thickness = float(airfoil["thickness"])

    if d.chord_tip_m > d.chord_root_m:
        d.chord_tip_m = d.chord_root_m
    if d.hub_radius_m >= d.radius_m * 0.8:
        d.hub_radius_m = 0.2 * d.radius_m
    return d


def objective_for_design(design: RotorDesign, env: Environment, body: Body, cfg: SimConfig) -> float:
    solidity = design.blades * 0.5 * (design.chord_root_m + design.chord_tip_m) / (np.pi * design.radius_m)
    penalty = 0.0
    if solidity > 0.35:
        penalty += 16.0 * (solidity - 0.35)
    if solidity < 0.04:
        penalty += 12.0 * (0.04 - solidity)

    result = simulate_drop(design, env, body, cfg)
    if result.max_rpm > 1800:
        penalty += 0.004 * (result.max_rpm - 1800)
    penalty += 0.12 * max(result.impact_speed_m_s - 6.0, 0.0)
    return -(result.fall_time_s - penalty)


def objective_from_summary(design: RotorDesign, summary: dict) -> float:
    solidity = design.blades * 0.5 * (design.chord_root_m + design.chord_tip_m) / (np.pi * design.radius_m)
    penalty = 0.0
    if solidity > 0.35:
        penalty += 16.0 * (solidity - 0.35)
    if solidity < 0.04:
        penalty += 12.0 * (0.04 - solidity)
    if float(summary["max_rpm"]) > 1800:
        penalty += 0.004 * (float(summary["max_rpm"]) - 1800)
    penalty += 0.12 * max(float(summary["impact_speed_m_s"]) - 6.0, 0.0)
    return -(float(summary["fall_time_s"]) - penalty)


def _make_polar_interpolants(polar_npz_path: str):
    global _INTERP_COUNTER
    _INTERP_COUNTER += 1
    data = np.load(polar_npz_path)
    alpha_deg = np.asarray(data["alpha_deg"], dtype=float)
    re_values = np.asarray(data["re"], dtype=float)
    cl = np.asarray(data["cl"], dtype=float)
    cd = np.asarray(data["cd"], dtype=float)
    cl_i = cs.interpolant(f"cl_i_{_INTERP_COUNTER}", "linear", [re_values, alpha_deg], cl.ravel(order="F"))
    cd_i = cs.interpolant(f"cd_i_{_INTERP_COUNTER}", "linear", [re_values, alpha_deg], cd.ravel(order="F"))
    return {
        "alpha_min": float(alpha_deg[0]),
        "alpha_max": float(alpha_deg[-1]),
        "re_min": float(re_values[0]),
        "re_max": float(re_values[-1]),
        "cl": cl_i,
        "cd": cd_i,
    }


def _casadi_clip(x, lower, upper):
    return cs.fmin(cs.fmax(x, lower), upper)


def _prandtl_loss_cs(blades, r, radius, rhub, phi):
    sphi = cs.fmax(cs.fabs(cs.sin(phi)), 1e-5)
    f_tip = (blades / 2.0) * (radius - r) / (r * sphi + 1e-6)
    f_hub = (blades / 2.0) * (r - rhub) / (r * sphi + 1e-6)
    f_tip = cs.fmax(f_tip, 0.0)
    f_hub = cs.fmax(f_hub, 0.0)
    f_tip_val = (2.0/np.pi)*cs.acos(cs.exp(-f_tip))
    f_hub_val = (2.0/np.pi)*cs.acos(cs.exp(-f_hub))
    return _casadi_clip(f_tip_val * f_hub_val, 1e-3, 1.0)


def _station_loads_cs(r, radius, rhub, chord, beta, omega, v_down, polar, design: RotorDesign, env: Environment, cfg: SimConfig):
    sigma = design.blades * chord / (2.0 * np.pi * r)
    a = 0.10
    ap = 0.00
    for _ in range(cfg.induction_max_iter):
        va = cs.fmax(v_down, 0.0) * (1.0 + a)
        vt = cs.fmax(omega, 0.0) * r * (1.0 - ap)
        w = cs.sqrt(va * va + vt * vt + cfg.min_speed_eps**2)
        phi = cs.atan2(va, vt + 1e-9)
        alpha = phi - beta
        alpha_deg = _casadi_clip(alpha * 180.0 / np.pi, polar["alpha_min"], polar["alpha_max"])
        re = _casadi_clip(env.rho * w * chord / env.mu, polar["re_min"], polar["re_max"])
        cl = polar["cl"](cs.vertcat(re, alpha_deg))
        cd = cs.fmax(polar["cd"](cs.vertcat(re, alpha_deg)), 1e-4)

        cn = cl * cs.cos(phi) + cd * cs.sin(phi)
        ct = cl * cs.sin(phi) - cd * cs.cos(phi)
        f_loss = _prandtl_loss_cs(design.blades, r, radius, rhub, phi)

        denom_a = (4.0 * f_loss * cs.sin(phi) ** 2) / (sigma * cn + 1e-8)
        a_new = _casadi_clip(1.0 / (denom_a + 1.0), 0.0, 0.95)

        denom_ap = (4.0 * f_loss * cs.sin(phi) * cs.cos(phi)) / (sigma * ct + 1e-8)
        ap_new = _casadi_clip(1.0 / (denom_ap - 1.0), -0.5, 0.7)

        a = (1.0 - cfg.induction_relax) * a + cfg.induction_relax * a_new
        ap = (1.0 - cfg.induction_relax) * ap + cfg.induction_relax * ap_new

    va = cs.fmax(v_down, 0.0) * (1.0 + a)
    vt = cs.fmax(omega, 0.0) * r * (1.0 - ap)
    w = cs.sqrt(va * va + vt * vt + cfg.min_speed_eps**2)
    phi = cs.atan2(va, vt + 1e-9)
    alpha = phi - beta
    alpha_deg = _casadi_clip(alpha * 180.0 / np.pi, polar["alpha_min"], polar["alpha_max"])
    re = _casadi_clip(env.rho * w * chord / env.mu, polar["re_min"], polar["re_max"])
    cl = polar["cl"](cs.vertcat(re, alpha_deg))
    cd = cs.fmax(polar["cd"](cs.vertcat(re, alpha_deg)), 1e-4)

    cn = cl * cs.cos(phi) + cd * cs.sin(phi)
    ct = cl * cs.sin(phi) - cd * cs.cos(phi)
    q = 0.5 * env.rho * w * w
    fn = design.blades * q * chord * cn
    ft = design.blades * q * chord * ct
    return fn, ft


def _section_material_area_expr(chord, design: RotorDesign):
    shell_area = design.shell_perimeter_factor * chord * design.wall_thickness_m
    spar_height = design.spar_height_fraction * design.thickness * chord
    spar_area = design.spar_count * spar_height * design.wall_thickness_m
    return shell_area + spar_area


def _rotor_inertia_expr(radius, chord_root, chord_tip, design: RotorDesign, n_span: int):
    xis = np.linspace(0.0, 1.0, n_span)
    inertia = 0.0
    rhub = design.hub_radius_m
    dr = (radius - rhub) / max(n_span - 1, 1)
    for xi in xis:
        r = rhub + xi * (radius - rhub)
        chord = chord_root + xi * (chord_tip - chord_root)
        section_area = _section_material_area_expr(chord, design)
        dm = design.blades * design.material_density_kg_m3 * section_area * dr
        inertia += dm * r**2
    return inertia


def _rotor_mass_expr(radius, chord_root, chord_tip, design: RotorDesign, n_span: int):
    xis = np.linspace(0.0, 1.0, n_span)
    mass = 0.0
    rhub = design.hub_radius_m
    dr = (radius - rhub) / max(n_span - 1, 1)
    for xi in xis:
        chord = chord_root + xi * (chord_tip - chord_root)
        section_area = _section_material_area_expr(chord, design)
        mass += design.blades * design.material_density_kg_m3 * section_area * dr
    return mass


def _aero_loads_expr(radius, chord_root, chord_tip, twist_root, twist_tip, pitch, omega, v_down, polar, design, env, cfg):
    xis = np.linspace(0.0, 1.0, cfg.n_span)
    fn_total = 0.0
    tq_total = 0.0
    rhub = design.hub_radius_m
    dr = (radius - rhub) / max(cfg.n_span - 1, 1)
    for xi in xis:
        r = rhub + xi * (radius - rhub)
        chord = chord_root + xi * (chord_tip - chord_root)
        beta_deg = pitch + twist_root + xi * (twist_tip - twist_root)
        beta = beta_deg * np.pi / 180.0
        fn, ft = _station_loads_cs(r, radius, rhub, chord, beta, omega, v_down, polar, design, env, cfg)
        fn_total += fn * dr
        tq_total += ft * r * dr
    return fn_total, tq_total


def _default_warm_start(n_nodes: int, cfg: SimConfig):
    return {
        "radius_m": 0.28,
        "chord_root_m": 0.06,
        "chord_tip_m": 0.04,
        "twist_root_deg": 16.0,
        "twist_tip_deg": 2.0,
        "pitch_collective_deg": 6.0,
        "t_final": 2.8,
        "h": np.linspace(cfg.drop_height_m, 0.0, n_nodes),
        "v": np.linspace(cfg.v0_down_m_s, 8.0, n_nodes),
        "omega": np.linspace(cfg.omega0_rad_s, 80.0, n_nodes),
    }


def _resample_guess(values: np.ndarray, n_nodes: int):
    old_x = np.linspace(0.0, 1.0, len(values))
    new_x = np.linspace(0.0, 1.0, n_nodes)
    return np.interp(new_x, old_x, values)


def optimize_geometry_for_airfoil(airfoil: dict[str, float], env: Environment, body: Body, cfg_opt: SimConfig, cfg_eval: SimConfig, iters: int, warm_start: dict | None = None, stage_name: str = "coarse") -> tuple[RotorDesign, object, dict]:
    if asb is None or cs is None or asb_np is None:
        raise RuntimeError("AeroSandbox and CasADi are required for direct-transcription optimization.")

    print(f"[optimize]   {stage_name} direct transcription for {airfoil['name']}")
    polar = _make_polar_interpolants(cfg_opt.polar_npz_path)
    design_stub = default_design()
    design_stub.camber = float(airfoil["camber"])
    design_stub.camber_pos = float(airfoil["camber_pos"])
    design_stub.thickness = float(airfoil["thickness"])

    n_nodes = max(8, 5 + iters)
    guess = _default_warm_start(n_nodes=n_nodes, cfg=cfg_opt)
    if warm_start is not None:
        for key in ["radius_m", "chord_root_m", "chord_tip_m", "twist_root_deg", "twist_tip_deg", "pitch_collective_deg", "t_final"]:
            if key in warm_start:
                guess[key] = float(warm_start[key])
        for key in ["h", "v", "omega"]:
            if key in warm_start:
                guess[key] = _resample_guess(np.asarray(warm_start[key], dtype=float), n_nodes)

    opti = asb.Opti()
    radius = opti.variable(init_guess=guess["radius_m"], scale=0.25, lower_bound=GEOMETRY_BOUNDS["radius_m"][0], upper_bound=GEOMETRY_BOUNDS["radius_m"][1])
    chord_root = opti.variable(init_guess=guess["chord_root_m"], scale=0.06, lower_bound=GEOMETRY_BOUNDS["chord_root_m"][0], upper_bound=GEOMETRY_BOUNDS["chord_root_m"][1])
    chord_tip = opti.variable(init_guess=guess["chord_tip_m"], scale=0.04, lower_bound=GEOMETRY_BOUNDS["chord_tip_m"][0], upper_bound=GEOMETRY_BOUNDS["chord_tip_m"][1])
    twist_root = opti.variable(init_guess=guess["twist_root_deg"], scale=15.0, lower_bound=GEOMETRY_BOUNDS["twist_root_deg"][0], upper_bound=GEOMETRY_BOUNDS["twist_root_deg"][1])
    twist_tip = opti.variable(init_guess=guess["twist_tip_deg"], scale=8.0, lower_bound=GEOMETRY_BOUNDS["twist_tip_deg"][0], upper_bound=GEOMETRY_BOUNDS["twist_tip_deg"][1])
    pitch = opti.variable(init_guess=guess["pitch_collective_deg"], scale=8.0, lower_bound=GEOMETRY_BOUNDS["pitch_collective_deg"][0], upper_bound=GEOMETRY_BOUNDS["pitch_collective_deg"][1])
    t_final = opti.variable(init_guess=guess["t_final"], scale=3.0, lower_bound=1.0, upper_bound=12.0)

    time_vec = asb_np.linspace(0.0, t_final, n_nodes)
    h = opti.variable(init_guess=guess["h"], n_vars=n_nodes, scale=cfg_opt.drop_height_m)
    v = opti.variable(init_guess=guess["v"], n_vars=n_nodes, scale=10.0)
    omega = opti.variable(init_guess=guess["omega"], n_vars=n_nodes, scale=100.0)

    opti.subject_to(chord_tip <= chord_root)
    opti.subject_to(h[0] == cfg_opt.drop_height_m)
    opti.subject_to(v[0] == cfg_opt.v0_down_m_s)
    opti.subject_to(omega[0] == cfg_opt.omega0_rad_s)
    opti.subject_to(h[-1] == 0.0)
    opti.subject_to(h >= 0.0)
    opti.subject_to(v >= 0.0)
    opti.subject_to(omega >= 0.0)

    rotor_mass_total = _rotor_mass_expr(radius, chord_root, chord_tip, design_stub, cfg_opt.n_span)
    total_mass = body.mass_kg + rotor_mass_total
    inertia = _rotor_inertia_expr(radius, chord_root, chord_tip, design_stub, cfg_opt.n_span)
    solidity = design_stub.blades * 0.5 * (chord_root + chord_tip) / (np.pi * radius)
    opti.subject_to(solidity >= 0.04)
    opti.subject_to(solidity <= 0.35)

    a_down_values = []
    domega_values = []
    rpm_values = []
    for k in range(n_nodes):
        thrust, torque = _aero_loads_expr(radius, chord_root, chord_tip, twist_root, twist_tip, pitch, omega[k], v[k], polar, design_stub, env, cfg_opt)
        drag_body = 0.5 * env.rho * body.cd_body * body.area_body_m2 * v[k] * cs.fabs(v[k])
        a_down = env.g - (thrust + drag_body) / total_mass
        domega = (torque - design_stub.friction_coef * omega[k] * cs.fabs(omega[k])) / inertia
        a_down_values.append(a_down)
        domega_values.append(domega)
        rpm_values.append(omega[k] * 60.0 / (2.0 * np.pi))

    opti.constrain_derivative(derivative=-v, variable=h, with_respect_to=time_vec, method="trapezoidal")
    opti.constrain_derivative(derivative=cs.vertcat(*a_down_values), variable=v, with_respect_to=time_vec, method="trapezoidal")
    opti.constrain_derivative(derivative=cs.vertcat(*domega_values), variable=omega, with_respect_to=time_vec, method="trapezoidal")

    impact_speed = v[-1]
    max_rpm = cs.mmax(cs.vertcat(*rpm_values))
    penalty = 0.12 * cs.fmax(impact_speed - 6.0, 0.0) + 0.004 * cs.fmax(max_rpm - 1800.0, 0.0)
    opti.minimize(-(t_final - penalty))
    sol = opti.solve(max_iter=1000, verbose=False, options={}, behavior_on_failure="return_last")
    try:
        print(f"[optimize]   opti status: {sol.stats()['return_status']}")
    except Exception:
        pass

    best_x = np.asarray([float(sol(radius)), float(sol(chord_root)), float(sol(chord_tip)), float(sol(twist_root)), float(sol(twist_tip)), float(sol(pitch))], dtype=float)
    best_design = vec_to_design(best_x, airfoil)
    best_result = simulate_drop(best_design, env, body, cfg_eval)
    warm_out = {
        "radius_m": float(sol(radius)),
        "chord_root_m": float(sol(chord_root)),
        "chord_tip_m": float(sol(chord_tip)),
        "twist_root_deg": float(sol(twist_root)),
        "twist_tip_deg": float(sol(twist_tip)),
        "pitch_collective_deg": float(sol(pitch)),
        "t_final": float(sol(t_final)),
        "h": np.asarray(sol(h), dtype=float).reshape(-1),
        "v": np.asarray(sol(v), dtype=float).reshape(-1),
        "omega": np.asarray(sol(omega), dtype=float).reshape(-1),
        "status": sol.stats().get("return_status", "unknown"),
    }
    return best_design, best_result, warm_out


def _airfoil_cache_path(cache_dir: Path, backend: str, airfoil: dict[str, float], re_n: int, alpha_n: int) -> Path:
    return cache_dir / f"{backend}_{airfoil['name']}_re{re_n}_a{alpha_n}.npz"


def _opt_cache_signature(airfoil: dict[str, float], stage_name: str, cfg_opt: SimConfig, args, iters: int) -> str:
    payload = "|".join(
        [
            stage_name,
            airfoil["name"],
            f"camber={airfoil['camber']:.4f}",
            f"camber_pos={airfoil['camber_pos']:.4f}",
            f"thickness={airfoil['thickness']:.4f}",
            f"backend={args.backend}",
            f"model_size={args.model_size}",
            f"iters={iters}",
            f"dt={cfg_opt.dt_s}",
            f"n_span={cfg_opt.n_span}",
            f"induction_max_iter={cfg_opt.induction_max_iter}",
            f"drop_height={cfg_opt.drop_height_m}",
            f"re_min={args.re_min}",
            f"re_max={args.re_max}",
            f"re_n={args.re_n}",
            f"alpha_min={args.alpha_min}",
            f"alpha_max={args.alpha_max}",
            f"alpha_n={args.alpha_n}",
        ]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _stage_cache_path(cache_dir: Path, airfoil: dict[str, float], stage_name: str, signature: str) -> Path:
    return cache_dir / f"{stage_name}_{airfoil['name']}_{signature}.npz"


def _save_stage_cache(cache_path: Path, airfoil: dict[str, float], design: RotorDesign, result, warm: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        airfoil_name=airfoil["name"],
        camber=float(airfoil["camber"]),
        camber_pos=float(airfoil["camber_pos"]),
        thickness=float(airfoil["thickness"]),
        radius_m=float(design.radius_m),
        chord_root_m=float(design.chord_root_m),
        chord_tip_m=float(design.chord_tip_m),
        twist_root_deg=float(design.twist_root_deg),
        twist_tip_deg=float(design.twist_tip_deg),
        pitch_collective_deg=float(design.pitch_collective_deg),
        fall_time_s=float(result.fall_time_s),
        impact_speed_m_s=float(result.impact_speed_m_s),
        max_rpm=float(result.max_rpm),
        warm_radius_m=float(warm["radius_m"]),
        warm_chord_root_m=float(warm["chord_root_m"]),
        warm_chord_tip_m=float(warm["chord_tip_m"]),
        warm_twist_root_deg=float(warm["twist_root_deg"]),
        warm_twist_tip_deg=float(warm["twist_tip_deg"]),
        warm_pitch_collective_deg=float(warm["pitch_collective_deg"]),
        warm_t_final=float(warm["t_final"]),
        warm_h=np.asarray(warm["h"], dtype=float),
        warm_v=np.asarray(warm["v"], dtype=float),
        warm_omega=np.asarray(warm["omega"], dtype=float),
        warm_status=str(warm.get("status", "unknown")),
    )


def _load_stage_cache(cache_path: Path, airfoil: dict[str, float]):
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=False)
    design = default_design()
    design.camber = float(airfoil["camber"])
    design.camber_pos = float(airfoil["camber_pos"])
    design.thickness = float(airfoil["thickness"])
    design.radius_m = float(data["radius_m"])
    design.chord_root_m = float(data["chord_root_m"])
    design.chord_tip_m = float(data["chord_tip_m"])
    design.twist_root_deg = float(data["twist_root_deg"])
    design.twist_tip_deg = float(data["twist_tip_deg"])
    design.pitch_collective_deg = float(data["pitch_collective_deg"])
    warm = {
        "radius_m": float(data["warm_radius_m"]),
        "chord_root_m": float(data["warm_chord_root_m"]),
        "chord_tip_m": float(data["warm_chord_tip_m"]),
        "twist_root_deg": float(data["warm_twist_root_deg"]),
        "twist_tip_deg": float(data["warm_twist_tip_deg"]),
        "pitch_collective_deg": float(data["warm_pitch_collective_deg"]),
        "t_final": float(data["warm_t_final"]),
        "h": np.asarray(data["warm_h"], dtype=float),
        "v": np.asarray(data["warm_v"], dtype=float),
        "omega": np.asarray(data["warm_omega"], dtype=float),
        "status": str(data["warm_status"]),
    }
    summary = {
        "fall_time_s": float(data["fall_time_s"]),
        "impact_speed_m_s": float(data["impact_speed_m_s"]),
        "max_rpm": float(data["max_rpm"]),
    }
    return {"design": design, "warm": warm, "summary": summary}


def _cached_result(summary: dict):
    return SimpleNamespace(
        fall_time_s=float(summary["fall_time_s"]),
        impact_speed_m_s=float(summary["impact_speed_m_s"]),
        max_rpm=float(summary["max_rpm"]),
    )


def build_airfoil_candidates(args) -> list[dict[str, float]]:
    cambers = parse_grid(args.camber_grid)
    camber_positions = parse_grid(args.camber_pos_grid)
    thicknesses = parse_grid(args.thickness_grid)
    candidates = []
    for camber in cambers:
        for camber_pos in camber_positions:
            for thickness in thicknesses:
                candidates.append(
                    {
                        "camber": camber,
                        "camber_pos": camber_pos,
                        "thickness": thickness,
                        "name": naca4_from_design(camber, camber_pos, thickness),
                    }
                )
    return candidates


def optimize(iters: int, seed: int, args):
    t0 = time.time()
    print("[optimize] configuring simulation and airfoil loop...")

    env, body = Environment(), Body()
    cfg_eval = SimConfig()
    cfg_coarse = SimConfig(
        drop_height_m=cfg_eval.drop_height_m,
        dt_s=0.03,
        t_max_s=12.0,
        n_span=5,
        omega0_rad_s=cfg_eval.omega0_rad_s,
        v0_down_m_s=cfg_eval.v0_down_m_s,
        min_speed_eps=cfg_eval.min_speed_eps,
        induction_relax=cfg_eval.induction_relax,
        induction_max_iter=4,
        max_tip_mach=cfg_eval.max_tip_mach,
    )
    cfg_refine = SimConfig(
        drop_height_m=cfg_eval.drop_height_m,
        dt_s=0.02,
        t_max_s=12.0,
        n_span=7,
        omega0_rad_s=cfg_eval.omega0_rad_s,
        v0_down_m_s=cfg_eval.v0_down_m_s,
        min_speed_eps=cfg_eval.min_speed_eps,
        induction_relax=cfg_eval.induction_relax,
        induction_max_iter=6,
        max_tip_mach=cfg_eval.max_tip_mach,
    )

    re_values = np.geomspace(args.re_min, args.re_max, args.re_n)
    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_n)
    polar_cache_root = Path(args.polar_cache_dir) if args.polar_cache_dir else Path(tempfile.gettempdir()) / "autorotation_polars"
    polar_cache_root.mkdir(parents=True, exist_ok=True)
    opt_cache_root = Path(args.opt_cache_dir)
    opt_cache_root.mkdir(parents=True, exist_ok=True)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    airfoils = build_airfoil_candidates(args)
    print(f"[optimize] outer loop: {len(airfoils)} airfoil candidates")

    coarse_records = []
    coarse_summaries = []
    previous_warm = None

    for idx, airfoil in enumerate(airfoils, start=1):
        print(f"[optimize] airfoil {idx}/{len(airfoils)}: {airfoil['name']}")
        polar_path = _airfoil_cache_path(polar_cache_root, args.backend, airfoil, args.re_n, args.alpha_n)
        if not polar_path.exists():
            generate_polar_npz(
                backend=args.backend,
                out_path=str(polar_path),
                camber=airfoil["camber"],
                camber_pos=airfoil["camber_pos"],
                thickness=airfoil["thickness"],
                re_values=re_values,
                alpha_values_deg=alpha_values,
                xfoil_command=args.xfoil_command,
                timeout_s=args.timeout,
                max_iter=args.max_iter,
                model_size=args.model_size,
            )
        cfg_coarse.polar_npz_path = str(polar_path)
        cfg_refine.polar_npz_path = str(polar_path)
        cfg_eval.polar_npz_path = str(polar_path)

        coarse_sig = _opt_cache_signature(airfoil=airfoil, stage_name="coarse", cfg_opt=cfg_coarse, args=args, iters=iters)
        coarse_cache_path = _stage_cache_path(opt_cache_root, airfoil, "coarse", coarse_sig)
        coarse_cached = None if args.no_opt_cache else _load_stage_cache(coarse_cache_path, airfoil)

        if coarse_cached is not None:
            design = coarse_cached["design"]
            warm = coarse_cached["warm"]
            coarse_status = warm.get("status", "cache")
            print(f"[optimize]   coarse cache hit for {airfoil['name']}")
            result = _cached_result(coarse_cached["summary"])
            score = objective_from_summary(design, coarse_cached["summary"])
            print(f"[optimize]   result: fall_time={result.fall_time_s:.3f}s impact={result.impact_speed_m_s:.3f}m/s")
        else:
            design, result, warm = optimize_geometry_for_airfoil(airfoil, env, body, cfg_coarse, cfg_eval, iters, warm_start=previous_warm, stage_name="coarse")
            score = objective_for_design(design, env, body, cfg_eval)
            coarse_status = warm.get("status", "unknown")
            print(f"[optimize]   result: fall_time={result.fall_time_s:.3f}s impact={result.impact_speed_m_s:.3f}m/s")
            if not args.no_opt_cache:
                _save_stage_cache(coarse_cache_path, airfoil, design, result, warm)

        coarse_summaries.append(summary_record(airfoil, design, result, score, "coarse", coarse_status))
        coarse_records.append({"airfoil": airfoil, "design": design, "result": result, "score": score, "warm": warm, "polar_path": str(polar_path)})
        previous_warm = warm

    coarse_records.sort(key=lambda rec: rec["score"])
    top_k = min(3, len(coarse_records))
    print(f"[optimize] refinement pass on top {top_k} airfoils...")

    best_airfoil = None
    best_design = None
    best_result = None
    best_score = np.inf
    previous_refine_warm = None
    refine_summaries = []

    for rank, rec in enumerate(coarse_records[:top_k], start=1):
        airfoil = rec["airfoil"]
        print(f"[optimize] refine {rank}/{top_k}: {airfoil['name']}")
        cfg_refine.polar_npz_path = rec["polar_path"]
        cfg_eval.polar_npz_path = rec["polar_path"]
        warm_start = previous_refine_warm if previous_refine_warm is not None else rec["warm"]

        refine_iters = max(iters + 1, 2)
        refine_sig = _opt_cache_signature(airfoil=airfoil, stage_name="refine", cfg_opt=cfg_refine, args=args, iters=refine_iters)
        refine_cache_path = _stage_cache_path(opt_cache_root, airfoil, "refine", refine_sig)
        refine_cached = None if args.no_opt_cache else _load_stage_cache(refine_cache_path, airfoil)

        if refine_cached is not None:
            design = refine_cached["design"]
            warm = refine_cached["warm"]
            refine_status = warm.get("status", "cache")
            print(f"[optimize]   refine cache hit for {airfoil['name']}")
            result = _cached_result(refine_cached["summary"])
            score = objective_from_summary(design, refine_cached["summary"])
            print(f"[optimize]   refined result: fall_time={result.fall_time_s:.3f}s impact={result.impact_speed_m_s:.3f}m/s")
        else:
            design, result, warm = optimize_geometry_for_airfoil(airfoil, env, body, cfg_refine, cfg_eval, refine_iters, warm_start=warm_start, stage_name="refine")
            score = objective_for_design(design, env, body, cfg_eval)
            refine_status = warm.get("status", "unknown")
            print(f"[optimize]   refined result: fall_time={result.fall_time_s:.3f}s impact={result.impact_speed_m_s:.3f}m/s")
            if not args.no_opt_cache:
                _save_stage_cache(refine_cache_path, airfoil, design, result, warm)

        previous_refine_warm = warm
        refine_summaries.append(summary_record(airfoil, design, result, score, "refine", refine_status))

        if score < best_score:
            best_airfoil = airfoil
            best_design = design
            best_result = result
            best_score = score

    if best_design is not None:
        best_result = simulate_drop(best_design, env, body, cfg_eval)

    combined_summaries = coarse_summaries + refine_summaries
    write_summary_csv(report_dir / "optimization_summary.csv", combined_summaries)
    written_plots = generate_report_plots(report_dir, coarse_summaries, refine_summaries, best_result)

    print("[optimize] top coarse candidates:")
    for rec in sorted(coarse_summaries, key=lambda rec: rec["score"])[: min(5, len(coarse_summaries))]:
        print(f"[optimize]   {rec['name']}: fall={rec['fall_time_s']:.3f}s impact={rec['impact_speed_m_s']:.3f}m/s rpm={rec['max_rpm']:.0f} status={rec['status']}")
    if refine_summaries:
        print("[optimize] refined finalists:")
        for rec in sorted(refine_summaries, key=lambda rec: rec["score"]):
            print(f"[optimize]   {rec['name']}: fall={rec['fall_time_s']:.3f}s impact={rec['impact_speed_m_s']:.3f}m/s rpm={rec['max_rpm']:.0f} status={rec['status']}")
    print(f"[optimize] summary CSV: {report_dir / 'optimization_summary.csv'}")
    if written_plots:
        print("[optimize] plots:")
        for path in written_plots:
            print(f"[optimize]   {path}")

    print(f"[optimize] done in {time.time() - t0:.1f}s")
    return best_airfoil, best_design, best_result
