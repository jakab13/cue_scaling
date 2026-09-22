from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CUE_ORDER = ["ILD", "ITD", "COMBINED"]

# Matplotlib default colors:
# C0 = blue, C1 = orange, C2 = green
CUE_COLORS = {
    "ILD": "C0",
    "ITD": "C1",
    "COMBINED": "C2",
}


def load_k_slope_tables(derivatives_root="data/psychometrics"):
    """
    Load point-level and slope-level k-slope tables.
    """

    derivatives_root = Path(derivatives_root)

    points_path = derivatives_root / "k_slope_points.csv"
    slopes_path = derivatives_root / "k_slope_summary.csv"

    if not points_path.exists():
        raise FileNotFoundError(
            f"No k-slope points table found at: {points_path}. "
            "Run estimate_k_slopes(...) first."
        )

    if not slopes_path.exists():
        raise FileNotFoundError(
            f"No k-slope summary table found at: {slopes_path}. "
            "Run estimate_k_slopes(...) first."
        )

    points = pd.read_csv(points_path)
    slopes = pd.read_csv(slopes_path)

    return points, slopes


def filter_k_slope_tables(
    points,
    slopes,
    subject_id=None,
    reference_cues=("ILD", "ITD", "COMBINED"),
    comparison_cue="ILD",
):
    """
    Filter k-slope point and summary tables for plotting.
    """

    points = points.copy()
    slopes = slopes.copy()

    reference_cues = [cue.upper() for cue in reference_cues]
    comparison_cue = comparison_cue.upper()

    for df in [points, slopes]:
        df["reference_cue"] = df["reference_cue"].astype(str).str.upper()
        df["comparison_cue"] = df["comparison_cue"].astype(str).str.upper()

    if subject_id is not None:
        points = points[points["subject_id"].astype(str) == str(subject_id)]
        slopes = slopes[slopes["subject_id"].astype(str) == str(subject_id)]

    points = points[points["reference_cue"].isin(reference_cues)]
    slopes = slopes[slopes["reference_cue"].isin(reference_cues)]

    points = points[points["comparison_cue"] == comparison_cue]
    slopes = slopes[slopes["comparison_cue"] == comparison_cue]

    numeric_point_cols = [
        "reference_angle_folded",
        "reference_center_frequency",
        "comparison_center_frequency",
        "PSE",
    ]

    for col in numeric_point_cols:
        if col in points.columns:
            points[col] = pd.to_numeric(points[col], errors="coerce")

    numeric_slope_cols = [
        "reference_center_frequency",
        "comparison_center_frequency",
        "k_slope",
        "sigma_y",
        "n_points",
    ]

    for col in numeric_slope_cols:
        if col in slopes.columns:
            slopes[col] = pd.to_numeric(slopes[col], errors="coerce")

    points = points.dropna(
        subset=[
            "reference_angle_folded",
            "reference_center_frequency",
            "PSE",
        ]
    )

    slopes = slopes.dropna(
        subset=[
            "reference_center_frequency",
            "k_slope",
        ]
    )

    return points, slopes


def plot_k_slopes(
    subject_id,
    derivatives_root="data/psychometrics",
    frequencies=None,
    reference_cues=("ILD", "ITD", "COMBINED"),
    comparison_cue="ILD",
    min_points_for_line=2,
    x_max=None,
    y_max=None,
    save=False,
    save_path=None,
):
    """
    Plot k-slope measurements for one subject.

    This function only plots. It does not estimate slopes.
    Run estimate_k_slopes(...) first to create:
        k_slope_points.csv
        k_slope_summary.csv
    """

    if frequencies is None:
        frequencies = [400, 600, 800, 1000, 1200, 1400]

    points, slopes = load_k_slope_tables(
        derivatives_root=derivatives_root,
    )

    points, slopes = filter_k_slope_tables(
        points=points,
        slopes=slopes,
        subject_id=subject_id,
        reference_cues=reference_cues,
        comparison_cue=comparison_cue,
    )

    # Always show requested/default frequencies.
    # Append extra frequencies if they exist in the data.
    existing_freqs = sorted(points["reference_center_frequency"].dropna().unique())
    ordered_freqs = [float(freq) for freq in frequencies]

    for freq in existing_freqs:
        freq = float(freq)
        if freq not in ordered_freqs:
            ordered_freqs.append(freq)

    ordered_freqs = sorted(ordered_freqs)

    n_freqs = len(ordered_freqs)

    # Global x/y limits
    if x_max is not None:
        global_x_max = float(x_max)
    elif points.empty:
        global_x_max = 30
    else:
        global_x_max = points["reference_angle_folded"].max()

    if y_max is not None:
        global_y_max = float(y_max)
    elif points.empty:
        global_y_max = 5
    else:
        global_y_max = points["PSE"].max()

        if "sigma_y" in slopes.columns:
            max_sigma = pd.to_numeric(
                slopes["sigma_y"],
                errors="coerce",
            ).max()

            if np.isfinite(max_sigma):
                global_y_max += max_sigma

    global_x_limit = global_x_max * 1.05
    global_y_limit = global_y_max * 1.05

    fig, axes = plt.subplots(
        1,
        n_freqs,
        figsize=(3.2 * n_freqs, 4),
        sharex=True,
        sharey=True,
    )

    if n_freqs == 1:
        axes = [axes]

    for ax, freq in zip(axes, ordered_freqs):

        freq_points = points[points["reference_center_frequency"] == freq]
        freq_slopes = slopes[slopes["reference_center_frequency"] == freq]

        if freq_points.empty:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                alpha=0.35,
                fontsize=12,
            )

        k_text_lines = []

        for cue in CUE_ORDER:

            if cue not in [c.upper() for c in reference_cues]:
                continue

            color = CUE_COLORS.get(cue, None)

            cue_points = freq_points[freq_points["reference_cue"] == cue].copy()
            cue_slopes = freq_slopes[freq_slopes["reference_cue"] == cue].copy()

            if cue_points.empty:
                k_text_lines.append((cue, "None", color))
                continue

            cue_points = cue_points.sort_values("reference_angle_folded")

            x = cue_points["reference_angle_folded"].to_numpy(dtype=float)
            y = cue_points["PSE"].to_numpy(dtype=float)

            # ------------------------------------------------------------
            # Cue-specific sigma_y
            # ------------------------------------------------------------

            if (
                    not cue_slopes.empty
                    and "sigma_y" in cue_slopes.columns
            ):
                sigma_y = cue_slopes["sigma_y"].iloc[0]
            else:
                sigma_y = np.nan

            # ------------------------------------------------------------
            # ± sigma_y around each PSE point
            # ------------------------------------------------------------

            if np.isfinite(sigma_y):
                ax.errorbar(
                    x,
                    y,
                    yerr=sigma_y,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.2,
                    alpha=0.45,
                    capsize=0,
                    zorder=1,
                )

            # PSE point itself
            ax.scatter(
                x,
                y,
                color=color,
                alpha=0.9,
                zorder=2,
            )

            if cue_slopes.empty:
                k = np.nan
                n_points = len(cue_points)
            else:
                k = cue_slopes["k_slope"].iloc[0]
                n_points = cue_slopes["n_points"].iloc[0]

            if np.isfinite(k):
                if np.isfinite(sigma_y):
                    text = f"k={k:.2f}, σ={sigma_y:.2f}"
                else:
                    text = f"k={k:.2f}"

                k_text_lines.append((cue, text, color))

            if np.isfinite(k) and n_points >= min_points_for_line:

                if k > 0:
                    line_x_max = min(global_x_max, global_y_max / k)
                else:
                    line_x_max = global_x_max

                line_x_max = max(0, line_x_max)

                x_line = np.linspace(0, line_x_max, 100)
                y_line = k * x_line

                ax.plot(
                    x_line,
                    y_line,
                    color=color,
                    linewidth=2,
                )

        for line_idx, (cue, k_text, color) in enumerate(k_text_lines):
            ax.text(
                0.03,
                0.97 - line_idx * 0.10,
                k_text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color=color,
            )

        ax.set_title(f"{int(freq)}Hz", fontsize=14, fontweight="bold")
        ax.set_xlabel("Reference angle (°)")
        ax.axline((0, 0), slope=0, linewidth=0.8, color="lightgray")
        ax.axvline(0, linewidth=0.8, alpha=0.3)
        ax.set_xlim(0, global_x_limit)
        ax.set_ylim(0, global_y_limit)

    axes[0].set_ylabel(f"{comparison_cue.upper()} PSE (dB)")

    fig.suptitle(
        f"k-slope estimates | subject: {subject_id}",
        y=0.95,
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    if save:
        if save_path is None:
            save_dir = Path(derivatives_root) / "figures" / str(subject_id)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{subject_id}_k_slopes.png"

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved k-slope plot to: {save_path}")

    return fig, axes