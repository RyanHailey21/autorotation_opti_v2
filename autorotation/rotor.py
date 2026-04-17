from dataclasses import dataclass
import numpy as np

try:
    from .aero import airfoil_coeffs
    from .models import RotorDesign, Environment, SimConfig
    from .polars import PolarLookup
except ImportError:
    from aero import airfoil_coeffs
    from models import RotorDesign, Environment, SimConfig
    from polars import PolarLookup


_POLAR_CACHE = {}


@dataclass
class RotorStateOutputs:
    thrust_up_n: float
    torque_nm: float
    mean_alpha_deg: float
    mean_re: float


def span_geometry(design: RotorDesign, n_span: int):
    r = np.linspace(design.hub_radius_m, design.radius_m, n_span)
    xi = (r - design.hub_radius_m) / max(design.radius_m - design.hub_radius_m, 1e-9)
    chord = design.chord_root_m + (design.chord_tip_m - design.chord_root_m) * xi
    beta_deg = design.pitch_collective_deg + design.twist_root_deg + (design.twist_tip_deg - design.twist_root_deg) * xi
    beta = np.deg2rad(beta_deg)
    return r, chord, beta


def _section_material_area(design: RotorDesign, chord: np.ndarray):
    shell_area = design.shell_perimeter_factor * chord * design.wall_thickness_m
    spar_height = design.spar_height_fraction * design.thickness * chord
    spar_area = design.spar_count * spar_height * design.wall_thickness_m
    return shell_area + spar_area


def rotor_mass(design: RotorDesign, n_span: int = 80):
    r, chord, _ = span_geometry(design, n_span)
    dr = np.gradient(r)
    section_area = _section_material_area(design, chord)
    dm = design.material_density_kg_m3 * section_area * dr
    return float(design.blades * np.sum(dm))


def rotor_inertia(design: RotorDesign, n_span: int = 80):
    r, chord, _ = span_geometry(design, n_span)
    dr = np.gradient(r)
    section_area = _section_material_area(design, chord)
    dm = design.material_density_kg_m3 * section_area * dr
    return float(design.blades * np.sum(dm * r**2))


def _prandtl_loss(B: int, r: float, R: float, Rhub: float, phi: float):
    sphi = max(abs(np.sin(phi)), 1e-5)
    f_tip = (B / 2.0) * (R - r) / max(r * sphi, 1e-6)
    f_hub = (B / 2.0) * (r - Rhub) / max(r * sphi, 1e-6)
    f_tip = max(f_tip, 0.0)
    f_hub = max(f_hub, 0.0)
    F_tip = (2.0 / np.pi) * np.arccos(np.exp(-f_tip))
    F_hub = (2.0 / np.pi) * np.arccos(np.exp(-f_hub))
    F = np.clip(F_tip * F_hub, 1e-3, 1.0)
    return F


def _solve_station_induction(
    r: float,
    chord: float,
    beta: float,
    omega: float,
    v_down: float,
    design: RotorDesign,
    env: Environment,
    cfg: SimConfig,
    polar_lookup,
):
    sigma = design.blades * chord / (2.0 * np.pi * r)
    a = 0.10
    ap = 0.00

    for _ in range(cfg.induction_max_iter):
        va = max(v_down, 0.0) * (1.0 + a)
        vt = max(omega, 0.0) * r * (1.0 - ap)
        w = np.sqrt(va * va + vt * vt + cfg.min_speed_eps**2)
        phi = np.arctan2(va, vt + 1e-9)
        alpha = phi - beta
        re = env.rho * w * chord / env.mu
        if polar_lookup is None:
            cl, cd = airfoil_coeffs(alpha, re, design.camber, design.thickness, design.camber_pos)
        else:
            cl, cd = polar_lookup.coeffs(alpha, np.asarray([re]))
            cl = float(np.asarray(cl).reshape(-1)[0])
            cd = float(np.asarray(cd).reshape(-1)[0])

        cn = cl * np.cos(phi) + cd * np.sin(phi)
        ct = cl * np.sin(phi) - cd * np.cos(phi)

        F = _prandtl_loss(design.blades, r, design.radius_m, design.hub_radius_m, phi)
        denom_a = (4.0 * F * np.sin(phi) ** 2) / max(sigma * cn, 1e-8)
        a_new = 1.0 / (denom_a + 1.0)
        a_new = float(np.clip(a_new, 0.0, 0.95))

        denom_ap = (4.0 * F * np.sin(phi) * np.cos(phi)) / max(sigma * ct, 1e-8)
        ap_new = 1.0 / (denom_ap - 1.0)
        ap_new = float(np.clip(ap_new, -0.5, 0.7))

        a = (1.0 - cfg.induction_relax) * a + cfg.induction_relax * a_new
        ap = (1.0 - cfg.induction_relax) * ap + cfg.induction_relax * ap_new

    va = max(v_down, 0.0) * (1.0 + a)
    vt = max(omega, 0.0) * r * (1.0 - ap)
    w = np.sqrt(va * va + vt * vt + cfg.min_speed_eps**2)
    phi = np.arctan2(va, vt + 1e-9)
    alpha = phi - beta
    re = env.rho * w * chord / env.mu
    if polar_lookup is None:
        cl, cd = airfoil_coeffs(alpha, re, design.camber, design.thickness, design.camber_pos)
    else:
        cl, cd = polar_lookup.coeffs(alpha, np.asarray([re]))
        cl = float(np.asarray(cl).reshape(-1)[0])
        cd = float(np.asarray(cd).reshape(-1)[0])

    cn = cl * np.cos(phi) + cd * np.sin(phi)
    ct = cl * np.sin(phi) - cd * np.cos(phi)

    q = 0.5 * env.rho * w * w
    fn_per_m = design.blades * q * chord * cn
    ft_per_m = design.blades * q * chord * ct
    return fn_per_m, ft_per_m, alpha, re


def aero_loads(design: RotorDesign, env: Environment, cfg: SimConfig, omega: float, v_down: float):
    r, chord, beta = span_geometry(design, cfg.n_span)
    dr = np.gradient(r)

    fn = np.zeros_like(r)
    ft = np.zeros_like(r)
    alpha = np.zeros_like(r)
    re = np.zeros_like(r)

    polar_lookup = None
    if cfg.polar_npz_path:
        if cfg.polar_npz_path not in _POLAR_CACHE:
            _POLAR_CACHE[cfg.polar_npz_path] = PolarLookup.from_npz(cfg.polar_npz_path)
        polar_lookup = _POLAR_CACHE[cfg.polar_npz_path]

    for i in range(len(r)):
        fn[i], ft[i], alpha[i], re[i] = _solve_station_induction(
            r=float(r[i]),
            chord=float(chord[i]),
            beta=float(beta[i]),
            omega=float(omega),
            v_down=float(v_down),
            design=design,
            env=env,
            cfg=cfg,
            polar_lookup=polar_lookup,
        )

    thrust_up = np.sum(fn * dr)
    torque = np.sum(ft * r * dr)
    return RotorStateOutputs(
        thrust_up_n=float(thrust_up),
        torque_nm=float(torque),
        mean_alpha_deg=float(np.rad2deg(np.mean(alpha))),
        mean_re=float(np.mean(re)),
    )
