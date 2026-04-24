from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from .rotor import rotor_mass
except ImportError:
    from rotor import rotor_mass


def summary_record(airfoil: dict, design, result, score: float, stage: str, status: str):
    return {
        "name": airfoil["name"],
        "camber": float(airfoil["camber"]),
        "camber_pos": float(airfoil["camber_pos"]),
        "thickness": float(airfoil["thickness"]),
        "fall_time_s": float(result.fall_time_s),
        "impact_speed_m_s": float(result.impact_speed_m_s),
        "max_rpm": float(result.max_rpm),
        "rotor_mass_g": round(rotor_mass(design) * 1000.0, 2),
        "radius_m": float(design.radius_m),
        "chord_root_m": float(design.chord_root_m),
        "chord_tip_m": float(design.chord_tip_m),
        "twist_root_deg": float(design.twist_root_deg),
        "twist_tip_deg": float(design.twist_tip_deg),
        "pitch_collective_deg": float(design.pitch_collective_deg),
        "score": float(score),
        "stage": stage,
        "status": status,
    }


def write_summary_csv(path: Path, records: list[dict]):
    if not records:
        return
    headers = [
        "stage",
        "name",
        "camber",
        "camber_pos",
        "thickness",
        "fall_time_s",
        "impact_speed_m_s",
        "max_rpm",
        "rotor_mass_g",
        "radius_m",
        "chord_root_m",
        "chord_tip_m",
        "twist_root_deg",
        "twist_tip_deg",
        "pitch_collective_deg",
        "score",
        "status",
    ]
    lines = [",".join(headers)]
    for rec in records:
        lines.append(",".join(str(rec[h]) for h in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report_plots(report_dir: Path, coarse_records: list[dict], refine_records: list[dict], best_result):
    if plt is None or best_result is None:
        return []

    written = []

    coarse_sorted = sorted(coarse_records, key=lambda rec: rec["score"])
    names = [rec["name"] for rec in coarse_sorted]
    fall_times = [rec["fall_time_s"] for rec in coarse_sorted]
    impact_speeds = [rec["impact_speed_m_s"] for rec in coarse_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, fall_times, color="#3A6EA5")
    ax.set_ylabel("Fall Time [s]")
    ax.set_title("Coarse Airfoil Screening")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    p = report_dir / "airfoil_fall_time_ranking.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(
        [rec["thickness"] for rec in coarse_sorted],
        [rec["camber"] for rec in coarse_sorted],
        c=fall_times,
        s=120,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.5,
    )
    for rec in coarse_sorted:
        ax.text(rec["thickness"] + 0.001, rec["camber"] + 0.0005, rec["name"], fontsize=8)
    ax.set_xlabel("Thickness")
    ax.set_ylabel("Camber")
    ax.set_title("Airfoil Design Space Colored by Fall Time")
    fig.colorbar(scatter, ax=ax, label="Fall Time [s]")
    fig.tight_layout()
    p = report_dir / "airfoil_design_space.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    if refine_records:
        refine_sorted = sorted(refine_records, key=lambda rec: rec["score"])
        top = refine_sorted[: min(3, len(refine_sorted))]
        labels = [rec["name"] for rec in top]
        x = np.arange(len(labels))
        width = 0.25
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.bar(x - width, [rec["fall_time_s"] for rec in top], width=width, label="Fall Time [s]", color="#2A9D8F")
        ax1.bar(x, [rec["impact_speed_m_s"] for rec in top], width=width, label="Impact Speed [m/s]", color="#E76F51")
        ax2.bar(x + width, [rec["rotor_mass_g"] for rec in top], width=width, label="Rotor Mass [g]", color="#F4A261", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("Fall Time [s] / Impact Speed [m/s]")
        ax2.set_ylabel("Rotor Mass [g]")
        ax1.set_title("Refined Finalists")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        p = report_dir / "refined_finalists.png"
        fig.savefig(p, dpi=180)
        plt.close(fig)
        written.append(p)

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(best_result.trace_t, best_result.trace_h, color="#264653")
    axes[0].set_ylabel("Height [m]")
    axes[0].set_title("Best Design Drop Trajectory")
    axes[1].plot(best_result.trace_t, best_result.trace_v, color="#E76F51")
    axes[1].set_ylabel("Descent Speed [m/s]")
    axes[2].plot(best_result.trace_t, best_result.trace_rpm, color="#2A9D8F")
    axes[2].set_ylabel("RPM")
    axes[2].set_xlabel("Time [s]")
    fig.tight_layout()
    p = report_dir / "best_trajectory.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(fall_times, impact_speeds, color="#7B2CBF", s=90)
    for rec in coarse_sorted:
        ax.text(rec["fall_time_s"] + 0.01, rec["impact_speed_m_s"] + 0.02, rec["name"], fontsize=8)
    ax.set_xlabel("Fall Time [s]")
    ax.set_ylabel("Impact Speed [m/s]")
    ax.set_title("Tradeoff: Fall Time vs Impact Speed")
    fig.tight_layout()
    p = report_dir / "fall_time_vs_impact_speed.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    masses_g = [rec["rotor_mass_g"] for rec in coarse_sorted]
    cambers = np.asarray([rec["camber"] for rec in coarse_sorted])
    unique_cambers = sorted(set(cambers.tolist()))
    cmap = plt.get_cmap("plasma", max(len(unique_cambers), 1))
    camber_colors = {c: cmap(i) for i, c in enumerate(unique_cambers)}

    fig, ax = plt.subplots(figsize=(9, 5))
    for rec, m in zip(coarse_sorted, masses_g):
        color = camber_colors[rec["camber"]]
        ax.scatter(m, rec["fall_time_s"], color=color, s=100, zorder=3)
        ax.text(m + 0.3, rec["fall_time_s"] + 0.05, rec["name"], fontsize=8)
    for c, color in camber_colors.items():
        ax.scatter([], [], color=color, label=f"camber={c:.0%}", s=80)
    ax.set_xlabel("Rotor Mass [g]")
    ax.set_ylabel("Fall Time [s]")
    ax.set_title("Mass vs Fall Time (Coarse, all airfoils)")
    ax.legend(title="Camber", loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = report_dir / "mass_vs_fall_time.png"
    fig.savefig(p, dpi=180)
    plt.close(fig)
    written.append(p)

    return written
