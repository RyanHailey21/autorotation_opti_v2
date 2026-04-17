from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Environment:
    rho: float = 1.225
    mu: float = 1.81e-5
    g: float = 9.80665


@dataclass
class Body:
    mass_kg: float = 0.2
    cd_body: float = 1.05
    area_body_m2: float = 0.018


@dataclass
class RotorDesign:
    blades: int = 2
    radius_m: float = 0.24
    hub_radius_m: float = 0.03
    chord_root_m: float = 0.06
    chord_tip_m: float = 0.03
    twist_root_deg: float = 18.0
    twist_tip_deg: float = 4.0
    pitch_collective_deg: float = 8.0
    camber: float = 0.04
    camber_pos: float = 0.4
    thickness: float = 0.11
    material_density_kg_m3: float = 0.65e3
    wall_thickness_m: float = 0.00045
    shell_perimeter_factor: float = 2.05
    spar_count: int = 2
    spar_height_fraction: float = 0.7
    friction_coef: float = 4.0e-6


@dataclass
class SimConfig:
    drop_height_m: float = 18.3
    dt_s: float = 0.01
    t_max_s: float = 25.0
    n_span: int = 20
    omega0_rad_s: float = 0.2
    v0_down_m_s: float = 0.0
    min_speed_eps: float = 0.1
    induction_relax: float = 0.25
    induction_max_iter: int = 35
    max_tip_mach: float = 0.35
    polar_npz_path: Optional[str] = None


@dataclass
class SimResult:
    fall_time_s: float
    impact_speed_m_s: float
    max_rpm: float
    mean_rpm_last_20pct: float
    trace_t: np.ndarray
    trace_h: np.ndarray
    trace_v: np.ndarray
    trace_rpm: np.ndarray
    trace_thrust_n: np.ndarray
    trace_torque_nm: np.ndarray
