# analysis/plot_jnds.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import psignifit as ps
import psignifit.psigniplot as psp


CUE_ORDER = ["COMBINED", "ITD", "ILD"]

CUE_COLORS = {
    "ILD": "C0",       # blue
    "ITD": "C1",       # orange
    "COMBINED": "C2",  # green
}


def load_psychometric_summary(derivatives_root="data/psychometrics"):
    """
    Load psychometric summary table.
    """

    summary_path = Path(derivatives_root) / "psychometric_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"No summary table found at: {summary_path}")

    return pd.read_csv(summary_path)


def filter_jnd_summary(
    summary,
    subject_id,
    cues=("COMBINED", "ITD", "ILD"),
):
    """
    Keep only centred within-cue JND fits.

    JND fits are defined as:
        fit_domain == angle
        reference_angle_folded == 0
        reference_cue == comparison_cue
    """

    df = summary.copy()

    required_cols = [
        "subject_id",
        "fit_domain",
        "reference_cue",
        "comparison_cue",
        "reference_angle_folded",
        "reference_center_frequency",
        "fit_json",
        "PSE",
        "threshold_84",
        "JND_84",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in summary table: {missing}")

    df = df[df["subject_id"].astype(str) == str(subject_id)]

    df = df[df["fit_domain"].astype(str).str.lower() == "angle"]

    df["reference_cue"] = df["reference_cue"].astype(str).str.upper()
    df["comparison_cue"] = df["comparison_cue"].astype(str).str.upper()

    df = df[df["reference_cue"] == df["comparison_cue"]]

    df["reference_angle_folded"] = pd.to_numeric(
        df["reference_angle_folded"],
        errors="coerce",
    )

    df = df[df["reference_angle_folded"] == 0]

    df = df[df["reference_cue"].isin([cue.upper() for cue in cues])]

    df["reference_center_frequency"] = pd.to_numeric(
        df["reference_center_frequency"],
        errors="coerce",
    )

    df["PSE"] = pd.to_numeric(df["PSE"], errors="coerce")
    df["threshold_84"] = pd.to_numeric(df["threshold_84"], errors="coerce")
    df["JND_84"] = pd.to_numeric(df["JND_84"], errors="coerce")

    return df.copy()


def draw_jnd_markers(
    ax,
    pse,
    threshold_84,
    color,
    bar_y=0.43,
):
    """
    Draw custom PSE, 84% threshold and JND bar.

    PSE marker:
        vertical line from y=0 to y=0.5

    84% marker:
        vertical line from y=0 to y=0.84

    JND bar:
        horizontal coloured bar from PSE to threshold_84
    """

    if np.isfinite(pse):
        ax.vlines(
            x=pse,
            ymin=0,
            ymax=0.5,
            color="lightgray",
            linewidth=1,
            alpha=0.5,
            zorder=0,
        )

    if np.isfinite(threshold_84):
        ax.vlines(
            x=threshold_84,
            ymin=0,
            ymax=0.84,
            color="lightgray",
            linewidth=1,
            alpha=0.5,
            zorder=0,
        )

    if np.isfinite(pse) and np.isfinite(threshold_84):
        ax.hlines(
            y=bar_y,
            xmin=pse,
            xmax=threshold_84,
            color=color,
            linewidth=4,
            alpha=0.9,
            zorder=3,
        )


def plot_jnds(
    subject_id,
    derivatives_root="data/psychometrics",
    frequencies=None,
    cues=("COMBINED", "ITD", "ILD"),
    save=False,
    save_path=None,
):
    """
    Plot centred JND psychometric fits.

    Columns are frequencies.
    Rows are cue types, ordered COMBINED, ITD, ILD.
    """

    if frequencies is None:
        frequencies = [400, 600, 800, 1000, 1200, 1400]

    summary = load_psychometric_summary(
        derivatives_root=derivatives_root,
    )

    df = filter_jnd_summary(
        summary=summary,
        subject_id=subject_id,
        cues=cues,
    )

    # Always show requested frequencies; append unexpected extra frequencies.
    existing_freqs = sorted(df["reference_center_frequency"].dropna().unique())
    ordered_freqs = [float(freq) for freq in frequencies]

    for freq in existing_freqs:
        freq = float(freq)
        if freq not in ordered_freqs:
            ordered_freqs.append(freq)

    ordered_cues = [cue for cue in CUE_ORDER if cue in [c.upper() for c in cues]]

    n_rows = len(ordered_cues)
    n_cols = len(ordered_freqs)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 2.4 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    observed_xlims = []

    for row_idx, cue in enumerate(ordered_cues):

        for col_idx, freq in enumerate(ordered_freqs):

            ax = axes[row_idx, col_idx]

            color = CUE_COLORS.get(cue, "C0")

            match = df[
                (df["reference_cue"] == cue)
                & (df["reference_center_frequency"] == freq)
            ]

            if match.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=11,
                    alpha=0.35,
                )

            else:
                # If there are duplicates, use the most recent row in the table.
                row = match.iloc[-1]

                fit_json = Path(row["fit_json"])

                if fit_json.exists():
                    result = ps.Result.load_json(fit_json)

                    psp.plot_psychometric_function(
                        result,
                        ax=ax,
                        plot_parameter=False,
                        x_label="",
                        y_label="",
                        data_color=color,
                        line_color=color,
                        data_size=0.1,
                        extrapolate_stimulus=0,
                    )

                    observed_xlims.append(ax.get_xlim())

                else:
                    ax.text(
                        0.5,
                        0.5,
                        "fit file\nmissing",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                        fontsize=10,
                        alpha=0.45,
                    )

                pse = row["PSE"]
                threshold_84 = row["threshold_84"]
                jnd_84 = row["JND_84"]

                if np.isfinite(pse) and np.isfinite(threshold_84):
                    draw_jnd_markers(
                        ax=ax,
                        pse=pse,
                        threshold_84=threshold_84,
                        color=color,
                        bar_y=0.43,
                    )

                if np.isfinite(jnd_84):
                    ax.text(
                        0.04,
                        0.92,
                        f"JND={jnd_84:.2f}°",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        color=color,
                    )
                else:
                    ax.text(
                        0.04,
                        0.92,
                        "JND=None",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        color=color,
                    )

            if row_idx == 0:
                ax.set_title(
                    f"{int(freq)}Hz",
                    fontsize=14,
                    fontweight="bold",
                )

            if col_idx == 0:
                ax.set_ylabel(cue, fontsize=12, fontweight="bold")

            if row_idx == n_rows - 1:
                ax.set_xlabel("Comparison angle (°)")

            ax.set_ylim(0, 1.05)

    # Shared x-axis based on all plotted psychometric functions
    if observed_xlims:
        global_x_min = min(xlim[0] for xlim in observed_xlims)
        global_x_max = max(xlim[1] for xlim in observed_xlims)
    else:
        global_x_min = -20
        global_x_max = 20

    # Make x-limits symmetric around zero for easier comparison.
    max_abs_x = max(abs(global_x_min), abs(global_x_max))
    global_x_min = -max_abs_x
    global_x_max = max_abs_x

    for ax in axes.ravel():
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(0, 1.05)

    fig.suptitle(
        f"JND psychometric fits | subject: {subject_id}",
        y=0.98,
        fontsize=16,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if save:
        if save_path is None:
            save_dir = Path(derivatives_root) / "figures" / str(subject_id)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{subject_id}_jnds.png"

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved JND plot to: {save_path}")

    return fig, axes