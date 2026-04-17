import argparse
from pathlib import Path
import time
import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator
except Exception:  # pragma: no cover
    RegularGridInterpolator = None


def naca4_from_design(camber: float, camber_pos: float, thickness: float) -> str:
    m = int(np.clip(np.round(camber * 100.0), 0, 9))
    p = int(np.clip(np.round(camber_pos * 10.0), 1, 9))
    t = int(np.clip(np.round(thickness * 100.0), 1, 40))
    return f"naca{m}{p}{t:02d}"


class PolarLookup:
    def __init__(self, alpha_deg: np.ndarray, re: np.ndarray, cl: np.ndarray, cd: np.ndarray):
        self.alpha_deg = np.asarray(alpha_deg, dtype=float)
        self.re = np.asarray(re, dtype=float)
        self.cl = np.asarray(cl, dtype=float)
        self.cd = np.asarray(cd, dtype=float)
        self._ok = RegularGridInterpolator is not None
        if self._ok:
            self._cl_i = RegularGridInterpolator((self.re, self.alpha_deg), self.cl, bounds_error=False, fill_value=None)
            self._cd_i = RegularGridInterpolator((self.re, self.alpha_deg), self.cd, bounds_error=False, fill_value=None)

    @classmethod
    def from_npz(cls, path: str):
        data = np.load(path)
        return cls(alpha_deg=data["alpha_deg"], re=data["re"], cl=data["cl"], cd=data["cd"])

    def coeffs(self, alpha_rad: np.ndarray, re: np.ndarray):
        if not self._ok:
            raise RuntimeError("scipy is required for polar interpolation.")
        alpha_deg = np.rad2deg(alpha_rad)
        re_c = np.clip(re, self.re[0], self.re[-1])
        a_c = np.clip(alpha_deg, self.alpha_deg[0], self.alpha_deg[-1])
        pts = np.column_stack([re_c.ravel(), a_c.ravel()])
        cl = self._cl_i(pts).reshape(np.shape(alpha_rad))
        cd = self._cd_i(pts).reshape(np.shape(alpha_rad))
        return cl, np.maximum(cd, 1e-4)


def generate_xfoil_npz(
    out_path: str,
    camber: float,
    camber_pos: float,
    thickness: float,
    re_values: np.ndarray,
    alpha_values_deg: np.ndarray,
    xfoil_command: str = "xfoil",
    timeout_s: float = 120.0,
    max_iter: int = 60,
):
    try:
        import aerosandbox as asb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("aerosandbox is not installed.") from exc

    from shutil import which

    if which(xfoil_command) is None:
        raise RuntimeError(f"XFoil executable '{xfoil_command}' was not found on PATH.")

    t0 = time.time()
    print(f"[polars] starting XFoil sweep for {len(re_values)} Reynolds points...")

    af_name = naca4_from_design(camber=camber, camber_pos=camber_pos, thickness=thickness)
    airfoil = asb.Airfoil(af_name)

    re_values = np.asarray(re_values, dtype=float)
    alpha_values_deg = np.asarray(alpha_values_deg, dtype=float)
    cl_grid = np.zeros((len(re_values), len(alpha_values_deg)))
    cd_grid = np.zeros((len(re_values), len(alpha_values_deg)))

    for i, re in enumerate(re_values):
        if i == 0 or (i + 1) % max(1, len(re_values) // 6) == 0 or (i + 1) == len(re_values):
            print(f"[polars] Re sweep {i + 1}/{len(re_values)} (Re={re:.0f})")

        def run_alpha(alpha_vec):
            xf = asb.XFoil(
                airfoil=airfoil,
                Re=float(re),
                xfoil_command=xfoil_command,
                max_iter=max_iter,
                timeout=timeout_s,
            )
            return xf.alpha(alpha=alpha_vec)

        got_valid = False
        for shrink in [1.0, 0.8, 0.6]:
            if shrink < 1.0:
                a_mid = 0.5 * (alpha_values_deg[0] + alpha_values_deg[-1])
                a_half = 0.5 * (alpha_values_deg[-1] - alpha_values_deg[0]) * shrink
                alpha_try = np.linspace(a_mid - a_half, a_mid + a_half, len(alpha_values_deg))
                print(f"[polars] retry at Re={re:.0f} with narrower alpha span ({shrink:.1f}x)")
            else:
                alpha_try = alpha_values_deg

            try:
                res = run_alpha(alpha_try)
            except Exception:
                # Fallback: split sweep into two chunks to improve convergence.
                try:
                    mid = len(alpha_try) // 2
                    a1 = alpha_try[: mid + 1]
                    a2 = alpha_try[mid:]
                    r1 = run_alpha(a1)
                    r2 = run_alpha(a2)
                    res = {
                        "alpha": np.concatenate([np.asarray(r1["alpha"]), np.asarray(r2["alpha"])]),
                        "CL": np.concatenate([np.asarray(r1["CL"]), np.asarray(r2["CL"])]),
                        "CD": np.concatenate([np.asarray(r1["CD"]), np.asarray(r2["CD"])]),
                    }
                except Exception:
                    continue

            a_raw = np.asarray(res["alpha"], dtype=float)
            cl_raw = np.asarray(res["CL"], dtype=float)
            cd_raw = np.asarray(res["CD"], dtype=float)
            m = np.isfinite(a_raw) & np.isfinite(cl_raw) & np.isfinite(cd_raw)
            if np.count_nonzero(m) >= 2:
                got_valid = True
                break

        if not got_valid:
            raise RuntimeError(
                f"XFoil failed to produce valid points at Re={re:.1f}. "
                "Try a thicker airfoil, lower alpha range, or higher timeout."
            )

        a_ok = a_raw[m]
        cl_ok = cl_raw[m]
        cd_ok = np.maximum(cd_raw[m], 1e-4)

        # Sort and collapse duplicate alpha entries before interpolation.
        idx = np.argsort(a_ok)
        a_ok = a_ok[idx]
        cl_ok = cl_ok[idx]
        cd_ok = cd_ok[idx]
        a_u, u_idx = np.unique(a_ok, return_index=True)
        cl_u = cl_ok[u_idx]
        cd_u = cd_ok[u_idx]

        cl_grid[i, :] = np.interp(alpha_values_deg, a_u, cl_u)
        cd_grid[i, :] = np.maximum(np.interp(alpha_values_deg, a_u, cd_u), 1e-4)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        alpha_deg=alpha_values_deg,
        re=re_values,
        cl=cl_grid,
        cd=cd_grid,
        naca=af_name,
        camber=float(camber),
        camber_pos=float(camber_pos),
        thickness=float(thickness),
    )
    print(f"[polars] done in {time.time() - t0:.1f}s")
    return out_path


def generate_neuralfoil_npz(
    out_path: str,
    camber: float,
    camber_pos: float,
    thickness: float,
    re_values: np.ndarray,
    alpha_values_deg: np.ndarray,
    model_size: str = "medium",
):
    try:
        import aerosandbox as asb
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("aerosandbox is required for NeuralFoil backend.") from exc

    t0 = time.time()
    print(f"[polars] starting NeuralFoil sweep for {len(re_values)} Reynolds points...")

    af_name = naca4_from_design(camber=camber, camber_pos=camber_pos, thickness=thickness)
    airfoil = asb.Airfoil(af_name)

    re_values = np.asarray(re_values, dtype=float)
    alpha_values_deg = np.asarray(alpha_values_deg, dtype=float)
    cl_grid = np.zeros((len(re_values), len(alpha_values_deg)))
    cd_grid = np.zeros((len(re_values), len(alpha_values_deg)))

    for i, re in enumerate(re_values):
        if i == 0 or (i + 1) % max(1, len(re_values) // 6) == 0 or (i + 1) == len(re_values):
            print(f"[polars] Re sweep {i + 1}/{len(re_values)} (Re={re:.0f})")

        aero = airfoil.get_aero_from_neuralfoil(
            alpha=alpha_values_deg,
            Re=float(re),
            mach=0.0,
            model_size=model_size,
        )
        cl_grid[i, :] = np.asarray(aero["CL"], dtype=float)
        cd_grid[i, :] = np.maximum(np.asarray(aero["CD"], dtype=float), 1e-4)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        alpha_deg=alpha_values_deg,
        re=re_values,
        cl=cl_grid,
        cd=cd_grid,
        naca=af_name,
        camber=float(camber),
        camber_pos=float(camber_pos),
        thickness=float(thickness),
        backend="neuralfoil",
        model_size=model_size,
    )
    print(f"[polars] done in {time.time() - t0:.1f}s")
    return out_path


def generate_polar_npz(
    backend: str,
    out_path: str,
    camber: float,
    camber_pos: float,
    thickness: float,
    re_values: np.ndarray,
    alpha_values_deg: np.ndarray,
    xfoil_command: str = "xfoil",
    timeout_s: float = 120.0,
    max_iter: int = 60,
    model_size: str = "default",
):
    if backend == "xfoil":
        return generate_xfoil_npz(
            out_path=out_path,
            camber=camber,
            camber_pos=camber_pos,
            thickness=thickness,
            re_values=re_values,
            alpha_values_deg=alpha_values_deg,
            xfoil_command=xfoil_command,
            timeout_s=timeout_s,
            max_iter=max_iter,
        )
    if backend == "neuralfoil":
        return generate_neuralfoil_npz(
            out_path=out_path,
            camber=camber,
            camber_pos=camber_pos,
            thickness=thickness,
            re_values=re_values,
            alpha_values_deg=alpha_values_deg,
            model_size=model_size,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _parse():
    p = argparse.ArgumentParser(description="Generate airfoil polar database (.npz) for rotor simulation.")
    p.add_argument("--out", required=True, help="Output npz path.")
    p.add_argument("--backend", choices=["xfoil", "neuralfoil"], default="neuralfoil")
    p.add_argument("--camber", type=float, default=0.04)
    p.add_argument("--camber-pos", type=float, default=0.4)
    p.add_argument("--thickness", type=float, default=0.12)
    p.add_argument("--xfoil-command", default="xfoil")
    p.add_argument("--timeout", type=float, default=120.0, help="XFoil timeout per sweep in seconds.")
    p.add_argument("--max-iter", type=int, default=60, help="XFoil max iterations.")
    p.add_argument("--model-size", type=str, default="medium", help="NeuralFoil model size (e.g. small/medium/large/xlarge).")
    p.add_argument("--re-min", type=float, default=2.0e4)
    p.add_argument("--re-max", type=float, default=5.0e5)
    p.add_argument("--re-n", type=int, default=12)
    p.add_argument("--alpha-min", type=float, default=-20.0)
    p.add_argument("--alpha-max", type=float, default=28.0)
    p.add_argument("--alpha-n", type=int, default=121)
    return p.parse_args()


def main():
    args = _parse()
    re_values = np.geomspace(args.re_min, args.re_max, args.re_n)
    alpha_values = np.linspace(args.alpha_min, args.alpha_max, args.alpha_n)
    path = generate_polar_npz(
        backend=args.backend,
        out_path=args.out,
        camber=args.camber,
        camber_pos=args.camber_pos,
        thickness=args.thickness,
        re_values=re_values,
        alpha_values_deg=alpha_values,
        xfoil_command=args.xfoil_command,
        timeout_s=args.timeout,
        max_iter=args.max_iter,
        model_size=args.model_size,
    )
    print(f"wrote_polar_npz={path}")


if __name__ == "__main__":
    main()
