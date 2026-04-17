import numpy as np


def _viterna_blend(alpha: np.ndarray, cl_lin: np.ndarray, alpha_stall: float, cd90: float):
    """
    Blend linear-region polars to a Viterna-style post-stall model.
    """
    sa = np.sin(alpha)
    ca = np.cos(alpha)
    cl_post = 0.5 * cd90 * np.sin(2.0 * alpha)
    cd_post = cd90 * sa * sa

    s = np.clip((np.abs(alpha) - alpha_stall) / np.deg2rad(8.0), 0.0, 1.0)
    cl = (1.0 - s) * cl_lin + s * cl_post
    return cl, cd_post, s


def airfoil_coeffs(alpha_rad: np.ndarray, re: np.ndarray, camber: float, thickness: float, camber_pos: float):
    """
    Reynolds-aware section model with post-stall blending.
    Intended for preliminary design when measured/XFoil polars are unavailable.
    """
    alpha = np.asarray(alpha_rad)
    re = np.maximum(np.asarray(re), 1e3)

    # Thin-airfoil-like slope corrected for thickness effects.
    cl_alpha = 2.0 * np.pi * (0.90 - 1.2 * (thickness - 0.11) ** 2)
    cl0 = 0.65 * camber * (1.0 + 0.2 * (0.45 - np.clip(camber_pos, 0.2, 0.8)))

    alpha_eff = alpha + cl0 / max(cl_alpha, 1e-9)
    alpha_stall = np.deg2rad(11.0 + 55.0 * (0.12 - abs(thickness - 0.12)))
    alpha_stall = np.clip(alpha_stall, np.deg2rad(9.0), np.deg2rad(18.0))

    cl_lin = cl_alpha * alpha_eff
    cd90 = 1.90 - 6.0 * (thickness - 0.12) ** 2
    cd90 = float(np.clip(cd90, 1.35, 2.05))

    cl, cd_post, s_post = _viterna_blend(alpha_eff, cl_lin, alpha_stall, cd90)

    # Low-Re drag model with profile drag rise.
    re_ref = 2.0e5
    re_factor = (re_ref / re) ** 0.28
    re_factor = np.clip(re_factor, 0.45, 3.5)
    cd0 = (0.008 + 0.22 * (thickness - 0.10) ** 2 + 0.03 * camber**2) * re_factor
    k = 0.012 + 0.03 * max(thickness, 0.04)
    cd_attached = cd0 + k * cl**2 + 0.01 * np.maximum(np.abs(alpha_eff) - np.deg2rad(6.0), 0.0) ** 1.3
    cd = (1.0 - s_post) * cd_attached + s_post * np.maximum(cd_post, cd_attached)

    return cl, cd
