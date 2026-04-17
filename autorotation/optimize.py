import argparse
from dataclasses import asdict

try:
    from .optimizer import optimize
except ImportError:
    from optimizer import optimize


def parse_args():
    p = argparse.ArgumentParser(description="Optimize passive autorotation rotor using an outer loop over airfoils.")
    p.add_argument("--iters", type=int, default=3, help="Inner geometry optimization rounds per airfoil.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--backend", choices=["neuralfoil", "xfoil"], default="neuralfoil")
    p.add_argument("--model-size", type=str, default="medium", help="NeuralFoil model size.")
    p.add_argument("--xfoil-command", default="xfoil")
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-iter", type=int, default=60)
    p.add_argument("--camber-grid", type=str, default="0.00,0.02,0.04")
    p.add_argument("--camber-pos-grid", type=str, default="0.4,0.5")
    p.add_argument("--thickness-grid", type=str, default="0.08,0.12")
    p.add_argument("--re-min", type=float, default=2.0e4)
    p.add_argument("--re-max", type=float, default=5.0e5)
    p.add_argument("--re-n", type=int, default=12)
    p.add_argument("--alpha-min", type=float, default=-20.0)
    p.add_argument("--alpha-max", type=float, default=28.0)
    p.add_argument("--alpha-n", type=int, default=121)
    p.add_argument("--polar-cache-dir", type=str, default=".cache/polars")
    p.add_argument("--opt-cache-dir", type=str, default=".cache/optimize")
    p.add_argument("--no-opt-cache", action="store_true", help="Disable optimization result cache.")
    p.add_argument("--report-dir", type=str, default="outputs/reports")
    return p.parse_args()


def main():
    args = parse_args()
    best_airfoil, best_design, best_res = optimize(iters=args.iters, seed=args.seed, args=args)

    print("best_airfoil:")
    for key, value in best_airfoil.items():
        print(f"  {key}: {value}")
    print("best_design:")
    for key, value in asdict(best_design).items():
        print(f"  {key}: {value}")
    print(f"best_fall_time_s: {best_res.fall_time_s:.3f}")
    print(f"impact_speed_m_s: {best_res.impact_speed_m_s:.3f}")
    print(f"max_rpm: {best_res.max_rpm:.1f}")


if __name__ == "__main__":
    main()
