# frequency_alignment/run_experiment.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiment.experiment_pipeline import (
    run_experiment_from_conditions,
)

from analysis.fit_psychometrics import (
    fit_run_files,
)

from frequency_alignment.builders import (
    make_alignment_condition
)

from stimuli.sound_handler import (
    get_interaural_level_spectrum,
    ild_from_ils,
    ild_slope_at_zero_fit,
)


# ============================================================
# PROJECT PATHS
# ============================================================

# ============================================================
# PROJECT PATHS
# ============================================================

MAIN_PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

PREVIOUS_K_PATH = (
    MAIN_PROJECT_ROOT
    / "data"
    / "psychometrics"
    / "k_slope_summary.csv"
)

SUBPROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = (
    SUBPROJECT_ROOT
    / "data"
)

RAW_ROOT = (
    DATA_ROOT
    / "raw"
)

PSYCHOMETRICS_ROOT = (
    DATA_ROOT
    / "psychometrics"
)

SUMMARY_FIGURE_ROOT = (
    PSYCHOMETRICS_ROOT
    / "summary_figures"
)

PLANNING_ROOT = (
    DATA_ROOT
    / "planning"
)

COMBINED_CANDIDATES_PATH = (
    PLANNING_ROOT
    / "combined_candidates.csv"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

# Default number of repetitions per comparison value.
#
# Can still be overridden from the master file, e.g.:
#
# run_low_ild(
#     SUBJECT_ID,
#     n_repetitions=4,
# )
#

DEFAULT_N_REPETITIONS = 8

TAB20 = plt.get_cmap("tab20").colors

REFERENCE_CUE_COLORS = {
    "ILD": TAB20[0],       # dark blue
    "ITD": TAB20[2],       # dark orange
    "COMBINED": TAB20[4],  # dark green
}

COMPARISON_CUE_COLORS = {
    "ILD": TAB20[1],       # light blue
    "ITD": TAB20[3],       # light orange
    "COMBINED": TAB20[5],  # light green
}

def _get_mapping_k(
    cue,
    frequency,
):
    """
    Get the mapping slope k for one cue and frequency.

    ILD:
        calculated directly from the KEMAR interaural
        level spectrum.

    ITD / COMBINED:
        group-median k from the previous psychophysical
        experiment.

    Returns None if no estimate is available.
    """

    cue = str(
        cue
    ).upper()

    frequency = float(
        frequency
    )

    # --------------------------------------------------------
    # ILD
    # --------------------------------------------------------

    if cue == "ILD":

        k = ild_slope_at_zero_fit(
            freq_hz=frequency,
            return_abs=True,
        )

        if (
            k is None
            or not np.isfinite(k)
        ):
            return None

        return float(k)

    # --------------------------------------------------------
    # ITD / COMBINED
    # --------------------------------------------------------

    return _get_previous_median_k(
        cue=cue,
        frequency=frequency,
    )

def _get_previous_median_k(
    cue,
    frequency,
):
    """
    Get the group-median k slope from the previous experiment.

    These values are independent of the current
    frequency-alignment experiment.

    Returns None if no previous estimate exists.
    """

    if not PREVIOUS_K_PATH.exists():
        return None

    df = pd.read_csv(
        PREVIOUS_K_PATH
    )

    if "k_slope" not in df.columns:
        return None

    df = df.copy()

    df["reference_cue"] = (
        df["reference_cue"]
        .astype(str)
        .str.upper()
    )

    df["reference_center_frequency"] = pd.to_numeric(
        df["reference_center_frequency"],
        errors="coerce",
    )

    df["k_slope"] = pd.to_numeric(
        df["k_slope"],
        errors="coerce",
    )

    # Previous k estimates were expressed in the common
    # ILD comparison space.
    if "comparison_cue" in df.columns:

        comparison_cue = (
            df["comparison_cue"]
            .astype(str)
            .str.upper()
        )

        ild_rows = (
            comparison_cue
            == "ILD"
        )

        if ild_rows.any():
            df = df[
                ild_rows
            ]

    selected = df[
        (
            df["reference_cue"]
            == cue.upper()
        )
        & np.isclose(
            df["reference_center_frequency"],
            float(frequency),
        )
    ]

    values = (
        selected["k_slope"]
        .dropna()
    )

    if values.empty:
        return None

    return float(
        values.median()
    )

def _get_relative_mapping(
    fit_row,
):
    """
    Reconstruct relative cue-scaling functions from the
    observed PSE.

    The reference mapping is normalised to k_ref = 1.

    At the PSE:

        k_ref * x_ref = k_comp * x_pse

    therefore:

        k_comp = x_ref / x_pse

    This means that the mapping visualisation works even when
    no independent k-slope measurements are available.
    """

    reference_angle = float(
        fit_row["reference_angle_folded"]
    )

    pse = float(
        fit_row["PSE"]
    )

    k_reference = 1.0

    if (
        not np.isfinite(pse)
        or np.isclose(pse, 0)
    ):
        k_comparison = np.nan

    else:
        k_comparison = (
            reference_angle
            / pse
        )

    return {
        "reference_angle": reference_angle,
        "PSE": pse,
        "k_reference": k_reference,
        "k_comparison": k_comparison,
    }

def _predict_ild_alignment(
    reference_frequency,
    comparison_frequency,
    reference_angle,
    angle_max=35,
    resolution=0.05,
):
    """
    Calculate space- and ILD-based alignment predictions.

    Space prediction:
        same angle across frequencies

    Cue prediction:
        comparison angle producing the same ILD
        as the reference stimulus
    """

    ils = get_interaural_level_spectrum()

    # --------------------------------------------------------
    # ILD of the reference
    # --------------------------------------------------------

    reference_ild = abs(
        float(
            ild_from_ils(
                azis_deg=reference_angle,
                freq_hz=reference_frequency,
                ils_dict=ils,
            )
        )
    )

    # --------------------------------------------------------
    # Search comparison-frequency ILDs
    # --------------------------------------------------------

    angles = np.arange(
        0,
        angle_max + resolution,
        resolution,
    )

    comparison_ilds = np.abs(
        ild_from_ils(
            azis_deg=angles,
            freq_hz=comparison_frequency,
            ils_dict=ils,
        )
    )

    idx = np.argmin(
        np.abs(
            comparison_ilds
            - reference_ild
        )
    )

    cue_prediction = float(
        angles[idx]
    )

    comparison_ild_at_space = abs(
        float(
            ild_from_ils(
                azis_deg=reference_angle,
                freq_hz=comparison_frequency,
                ils_dict=ils,
            )
        )
    )

    return {
        "space_prediction": float(
            reference_angle
        ),

        "cue_prediction": cue_prediction,

        "reference_value": reference_ild,

        "space_value": comparison_ild_at_space,

        "cue_value": reference_ild,

        "prediction_label": "ILD",
    }

def _get_combined_k(
    frequency,
):
    """
    Retrieve the empirical combined-cue k slope
    for one frequency from the planning table.
    """

    if not COMBINED_CANDIDATES_PATH.exists():
        return None

    df = pd.read_csv(
        COMBINED_CANDIDATES_PATH
    )

    frequencies = []

    # Reference-side k values
    if {
        "reference_frequency",
        "k_reference",
    }.issubset(df.columns):

        reference = df[
            [
                "reference_frequency",
                "k_reference",
            ]
        ].copy()

        reference.columns = [
            "frequency",
            "k",
        ]

        frequencies.append(
            reference
        )

    # Comparison-side k values
    if {
        "comparison_frequency",
        "k_comparison",
    }.issubset(df.columns):

        comparison = df[
            [
                "comparison_frequency",
                "k_comparison",
            ]
        ].copy()

        comparison.columns = [
            "frequency",
            "k",
        ]

        frequencies.append(
            comparison
        )

    if not frequencies:
        return None

    lookup = pd.concat(
        frequencies,
        ignore_index=True,
    )

    lookup["frequency"] = pd.to_numeric(
        lookup["frequency"],
        errors="coerce",
    )

    lookup["k"] = pd.to_numeric(
        lookup["k"],
        errors="coerce",
    )

    matches = lookup[
        lookup["frequency"]
        == float(frequency)
    ]

    if matches.empty:
        return None

    return float(
        matches["k"].median()
    )

def _predict_combined_alignment(
    reference_frequency,
    comparison_frequency,
    reference_angle,
):
    """
    Calculate space- and scaling-based predictions
    for COMBINED cues.
    """

    k_reference = _get_combined_k(
        reference_frequency
    )

    k_comparison = _get_combined_k(
        comparison_frequency
    )

    # Space prediction is always available.
    space_prediction = float(
        reference_angle
    )

    if (
        k_reference is None
        or k_comparison is None
        or k_comparison <= 0
    ):
        return {
            "space_prediction":
                space_prediction,

            "cue_prediction":
                None,

            "reference_value":
                None,

            "space_value":
                None,

            "cue_value":
                None,

            "prediction_label":
                "Combined scaling",
        }

    scaling_prediction = (
        k_reference
        / k_comparison
        * reference_angle
    )

    return {
        "space_prediction":
            space_prediction,

        "cue_prediction":
            float(scaling_prediction),

        "reference_value":
            float(
                k_reference
                * reference_angle
            ),

        "space_value":
            float(
                k_comparison
                * reference_angle
            ),

        "cue_value":
            float(
                k_reference
                * reference_angle
            ),

        "prediction_label":
            "Combined scaling",
    }

def _get_k_source(
    cue,
):
    """
    Human-readable source of the mapping slope.
    """

    cue = str(
        cue
    ).upper()

    if cue == "ILD":
        return "KEMAR ILS"

    return "Previous group median"

def _calculate_predictions(
    fit_row,
):
    """
    Calculate independent predictions for one condition.

    Mapping slopes:
        ILD:
            derived directly from the KEMAR ILS.

        ITD / COMBINED:
            group-median slopes from the previous
            psychophysical experiment.

    Space prediction:
        same physical angle.

    Cue-based prediction:
        equal mapped value:

            k_ref * x_ref
            =
            k_comp * x_comp

        therefore:

            x_comp
            =
            (k_ref / k_comp) * x_ref
    """

    reference_cue = str(
        fit_row["reference_cue"]
    ).upper()

    comparison_cue = str(
        fit_row["comparison_cue"]
    ).upper()

    reference_frequency = float(
        fit_row[
            "reference_center_frequency"
        ]
    )

    comparison_frequency = float(
        fit_row[
            "comparison_center_frequency"
        ]
    )

    reference_angle = float(
        fit_row[
            "reference_angle_folded"
        ]
    )

    # --------------------------------------------------------
    # Independent mapping slopes
    # --------------------------------------------------------

    k_reference = _get_mapping_k(
        cue=reference_cue,
        frequency=reference_frequency,
    )

    k_comparison = _get_mapping_k(
        cue=comparison_cue,
        frequency=comparison_frequency,
    )

    # --------------------------------------------------------
    # Space-based prediction
    # --------------------------------------------------------

    space_prediction = (
        reference_angle
    )

    # --------------------------------------------------------
    # Cue-based prediction
    # --------------------------------------------------------

    cue_prediction = None

    if (
        k_reference is not None
        and k_comparison is not None
        and np.isfinite(k_reference)
        and np.isfinite(k_comparison)
        and k_comparison > 0
    ):

        cue_prediction = (
            k_reference
            / k_comparison
            * reference_angle
        )

    return {
        "space_prediction":
            float(space_prediction),

        "cue_prediction":
            (
                float(cue_prediction)
                if cue_prediction is not None
                else None
            ),

        "k_reference":
            k_reference,

        "k_comparison":
            k_comparison,

        "reference_k_source":
            _get_k_source(
                reference_cue
            ),

        "comparison_k_source":
            _get_k_source(
                comparison_cue
            ),
    }


def run_alignment_block(
    subject_id,
    cue,
    reference_frequency,
    comparison_frequency,
    reference_angle,
    comparison_angles,
    n_repetitions=DEFAULT_N_REPETITIONS,
):
    """
    Run one frequency-alignment block.
    """

    cue = cue.upper()

    condition_id = (
        f"{cue.lower()}_"
        f"{reference_frequency}to{comparison_frequency}_"
        f"ref{reference_angle}"
    )

    block_name = (
        f"{cue}: "
        f"{reference_frequency} → "
        f"{comparison_frequency} Hz"
    )

    condition = make_alignment_condition(
        condition_id=condition_id,
        reference_frequency=reference_frequency,
        comparison_frequency=comparison_frequency,
        cue=cue,
        reference_angle=reference_angle,
        comparison_angles=comparison_angles,
        n_repetitions=n_repetitions,
    )

    _run_block(
        subject_id=subject_id,
        conditions=[condition],
        block_name=block_name,
    )

# ============================================================
# PUBLIC ANALYSIS FUNCTIONS
# ============================================================

def fit_psychometrics(
    subject_id,
    overwrite=False,
):
    """
    Fit all available frequency-alignment data for one subject.

    The fitting uses the existing psychometric fitting machinery,
    but reads and writes only within the frequency_alignment
    project folders.

    This means that the data and fits remain completely separate
    from the original cue-scaling experiment.
    """

    subject_raw_dir = (
        RAW_ROOT
        / subject_id
    )

    csv_files = sorted(
        subject_raw_dir.rglob("*.csv")
    )

    if not csv_files:

        print()
        print(
            f"No raw frequency-alignment data found "
            f"for {subject_id}."
        )
        print(
            f"Looked in: {subject_raw_dir}"
        )

        return None

    print()
    print("=" * 60)
    print("FITTING PSYCHOMETRICS")
    print("=" * 60)

    print(
        f"Subject: {subject_id}"
    )

    print(
        f"Raw files found: {len(csv_files)}"
    )

    print()

    summary = fit_run_files(
        csv_paths=csv_files,
        derivatives_root=PSYCHOMETRICS_ROOT,
        overwrite=overwrite,
        analysis_label="frequency_alignment",
    )

    print()
    print(
        "Psychometric fitting finished."
    )

    return summary


def plot_results(
    subject_id,
    save=True,
    show=True,
):
    """
    Plot the currently available frequency-alignment results.

    For each completed block, the figure shows:

        dashed line:
            space-based prediction

        dotted line:
            cue/scaling-based prediction

        point:
            measured PSE

    Only blocks for which an angle-domain psychometric fit exists
    are plotted. This means the function can also be used after
    only one or two blocks have been completed.
    """

    summary = _load_subject_summary(
        subject_id=subject_id,
    )

    if summary is None:
        return None

    return _plot_alignment_summary(
        summary=summary,
        subject_id=subject_id,
        save=save,
        show=show,
    )


# ============================================================
# INTERNAL EXPERIMENT FUNCTIONS
# ============================================================

def _run_block(
    subject_id,
    conditions,
    block_name,
):
    """
    Internal helper for running one block.
    """

    RAW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print(block_name.upper())
    print("=" * 60)

    print(
        f"Subject: {subject_id}"
    )

    print()

    input(
        "Check participant ID and setup. "
        "Press Enter to begin..."
    )

    run_experiment_from_conditions(
        subject_id=subject_id,
        conditions=conditions,
        save_root=RAW_ROOT,
    )

    print()
    print(
        f"{block_name} block completed."
    )


# ============================================================
# INTERNAL ANALYSIS FUNCTIONS
# ============================================================

def _load_subject_summary(
    subject_id,
):
    """
    Load angle-domain psychometric fits for one subject.
    """

    summary_path = (
        PSYCHOMETRICS_ROOT
        / "psychometric_summary.csv"
    )

    if not summary_path.exists():

        print()
        print(
            "No psychometric summary found."
        )

        print(
            "Run fit_psychometrics() first."
        )

        print(
            f"Expected file: {summary_path}"
        )

        return None

    summary = pd.read_csv(
        summary_path
    )

    summary = _normalise_summary(
        summary
    )

    # --------------------------------------------------------
    # Subject
    # --------------------------------------------------------

    summary = summary[
        summary["subject_id"].astype(str)
        == str(subject_id)
    ].copy()

    if summary.empty:

        print()
        print(
            f"No psychometric fits found for "
            f"{subject_id}."
        )

        return None

    # --------------------------------------------------------
    # Angle-domain fits only
    # --------------------------------------------------------

    if "fit_domain" in summary.columns:

        summary = summary[
            summary["fit_domain"]
            .astype(str)
            .str.lower()
            == "angle"
        ].copy()

    if summary.empty:

        print()
        print(
            f"No angle-domain psychometric fits "
            f"found for {subject_id}."
        )

        return None

    return summary


def _normalise_summary(
    summary,
):
    """
    Make relevant psychometric-summary columns easy to compare.
    """

    summary = summary.copy()

    cue_columns = [
        "reference_cue",
        "comparison_cue",
    ]

    for column in cue_columns:

        if column in summary.columns:

            summary[column] = (
                summary[column]
                .astype(str)
                .str.upper()
            )

    numeric_columns = [
        "reference_center_frequency",
        "comparison_center_frequency",
        "reference_angle_folded",
        "PSE",
    ]

    for column in numeric_columns:

        if column in summary.columns:

            summary[column] = pd.to_numeric(
                summary[column],
                errors="coerce",
            )

    return summary


def _get_condition_fit(
    summary,
    prediction,
):
    """
    Find the psychometric fit corresponding to one experimental
    block.
    """

    condition = summary[
        (
            summary["reference_cue"]
            == prediction["cue"]
        )
        & (
            summary["comparison_cue"]
            == prediction["cue"]
        )
        & (
            summary["reference_center_frequency"]
            == prediction["reference_frequency"]
        )
        & (
            summary["comparison_center_frequency"]
            == prediction["comparison_frequency"]
        )
    ].copy()

    # Also match the reference angle if that column is available.
    if (
        "reference_angle_folded"
        in condition.columns
    ):

        condition = condition[
            condition["reference_angle_folded"]
            == prediction["reference_angle"]
        ].copy()

    if condition.empty:
        return None

    # Normally there should be one row.
    #
    # If several are present, use the latest row in the
    # psychometric summary rather than failing completely.
    return condition.iloc[-1]


# ============================================================
# INTERNAL PLOTTING
# ============================================================

def _plot_alignment_summary(
    summary,
    subject_id,
    save=True,
    show=True,
):
    """
    One row per fitted frequency-alignment condition.

    Left:
        reconstructed physical-to-perceived mappings

    Right:
        observed PSE versus model predictions
    """

    summary = summary.copy()

    condition_columns = [
        "reference_cue",
        "comparison_cue",
        "reference_center_frequency",
        "comparison_center_frequency",
        "reference_angle_folded",
    ]

    # One row per condition
    summary = (
        summary
        .dropna(
            subset=["PSE"]
        )
        .drop_duplicates(
            subset=condition_columns,
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    if summary.empty:

        print(
            f"No fitted alignment conditions "
            f"found for {subject_id}."
        )

        return None

    n_conditions = len(
        summary
    )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        n_conditions,
        2,
        figsize=(
            5,
            4 * n_conditions,
        ),
        width_ratios=(3, 1),
        squeeze=False,
    )

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------

    for row_index, (_, fit_row) in enumerate(
        summary.iterrows()
    ):

        predictions = (
            _calculate_predictions(
                fit_row
            )
        )

        mapping_ax = axes[
            row_index,
            0,
        ]

        prediction_ax = axes[
            row_index,
            1,
        ]

        # ----------------------------------------------------
        # Mapping plot
        # ----------------------------------------------------

        _plot_mapping_panel(
            ax=mapping_ax,
            fit_row=fit_row,
            predictions=predictions,
        )

        # ----------------------------------------------------
        # Prediction plot
        # ----------------------------------------------------

        _plot_prediction_panel(
            ax=prediction_ax,
            fit_row=fit_row,
            predictions=predictions,
        )

        # ----------------------------------------------------
        # Row title
        # ----------------------------------------------------

        reference_cue = str(
            fit_row[
                "reference_cue"
            ]
        ).upper()

        comparison_cue = str(
            fit_row[
                "comparison_cue"
            ]
        ).upper()

        ref_freq = int(
            fit_row[
                "reference_center_frequency"
            ]
        )

        comp_freq = int(
            fit_row[
                "comparison_center_frequency"
            ]
        )

        ref_angle = float(
            fit_row[
                "reference_angle_folded"
            ]
        )

        mapping_ax.set_title(
            f"{reference_cue} "
            f"{ref_freq} → "
            f"{comparison_cue} "
            f"{comp_freq} Hz "
            f"(ref {ref_angle:g}°)",
            fontsize=11,
        )

        prediction_ax.set_title(
            "ΔPSE",
            fontsize=11,
        )

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        f"Frequency alignment — {subject_id}",
        fontsize=14,
    )

    # prediction_legend = [
    #     Line2D(
    #         [0],
    #         [0],
    #         color="black",
    #         linestyle="--",
    #         linewidth=1.6,
    #         label="Space-based prediction",
    #     ),
    #
    #     Line2D(
    #         [0],
    #         [0],
    #         color="black",
    #         linestyle="-",
    #         linewidth=1.6,
    #         label="Cue-based prediction",
    #     ),
    # ]
    #
    # fig.legend(
    #     handles=prediction_legend,
    #     loc="upper right",
    #     frameon=False,
    # )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.97,
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if save:

        SUMMARY_FIGURE_ROOT.mkdir(
            parents=True,
            exist_ok=True,
        )

        png_path = (
            SUMMARY_FIGURE_ROOT
            / (
                f"{subject_id}"
                "_frequency_alignment.png"
            )
        )

        svg_path = (
            SUMMARY_FIGURE_ROOT
            / (
                f"{subject_id}"
                "_frequency_alignment.svg"
            )
        )

        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
        )

        fig.savefig(
            svg_path,
            bbox_inches="tight",
        )

        print(
            f"Saved summary figure to:\n"
            f"{png_path}"
        )

    if show:
        plt.show()

    else:
        plt.close(fig)

    return fig, axes


def _plot_mapping_panel(
    ax,
    fit_row,
    predictions,
):
    """
    Plot independently estimated cue-scaling functions.

    k values come exclusively from the previous experiment.
    The current PSE is shown on top of those functions but
    is never used to calculate their slopes.
    """

    reference_angle = float(
        fit_row[
            "reference_angle_folded"
        ]
    )

    pse = float(
        fit_row["PSE"]
    )

    reference_cue = str(
        fit_row["reference_cue"]
    ).upper()

    comparison_cue = str(
        fit_row["comparison_cue"]
    ).upper()

    reference_frequency = int(
        fit_row[
            "reference_center_frequency"
        ]
    )

    comparison_frequency = int(
        fit_row[
            "comparison_center_frequency"
        ]
    )

    k_reference = predictions[
        "k_reference"
    ]

    k_comparison = predictions[
        "k_comparison"
    ]

    reference_color = REFERENCE_CUE_COLORS.get(
        reference_cue,
        TAB20[0],
    )

    comparison_color = COMPARISON_CUE_COLORS.get(
        comparison_cue,
        TAB20[1],
    )

    # --------------------------------------------------------
    # X range
    # --------------------------------------------------------

    x_values = [
        reference_angle,
        pse,
    ]

    cue_prediction = predictions.get(
        "cue_prediction"
    )

    if cue_prediction is not None:
        x_values.append(
            cue_prediction
        )

    x_max = max(
        max(x_values) * 1.25,
        15,
    )

    x = np.linspace(
        0,
        x_max,
        200,
    )

    plotted_something = False

    # --------------------------------------------------------
    # Reference scaling function
    # --------------------------------------------------------

    if k_reference is not None:

        y_reference = (
            k_reference
            * x
        )

        ax.plot(
            x,
            y_reference,
            color=reference_color,
            linewidth=2.5,
            linestyle="-",
            label=(
                f"Reference "
                f"{reference_frequency} Hz"
            ),
        )

        reference_y = (
            k_reference
            * reference_angle
        )

        ax.scatter(
            reference_angle,
            reference_y,
            color=reference_color,
            s=70,
            marker="o",
            zorder=4,
        )

        ax.vlines(
            reference_angle,
            0,
            reference_y,
            color=reference_color,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        plotted_something = True

    # --------------------------------------------------------
    # Comparison scaling function
    # --------------------------------------------------------

    if k_comparison is not None:

        y_comparison = (
            k_comparison
            * x
        )

        ax.plot(
            x,
            y_comparison,
            color=comparison_color,
            linewidth=2.5,
            linestyle="-",
            label=(
                f"Comparison "
                f"{comparison_frequency} Hz "
            ),
        )

        comparison_y = (
            k_comparison
            * pse
        )

        ax.scatter(
            pse,
            comparison_y,
            color=comparison_color,
            s=70,
            marker="s",
            zorder=4,
        )

        ax.vlines(
            pse,
            0,
            comparison_y,
            color=comparison_color,
            linestyle=":",
            linewidth=1,
            alpha=0.6,
        )

        plotted_something = True

        cue_prediction = predictions.get("cue_prediction")
        space_prediction = predictions.get("space_prediction")

        if (
                k_reference is not None
                and k_comparison is not None
        ):
            reference_y = k_reference * reference_angle
            comparison_y_at_space = k_comparison * space_prediction

            # --------------------------------------------
            # cue-based alignment:
            # same perceived location (horizontal line)
            # --------------------------------------------
            if cue_prediction is not None and np.isfinite(cue_prediction):
                ax.hlines(
                    y=reference_y,
                    xmin=min(reference_angle, cue_prediction),
                    xmax=max(reference_angle, cue_prediction),
                    color="black",
                    linestyle="-",
                    linewidth=1.5,
                    zorder=2,
                )

            # --------------------------------------------
            # space-based alignment:
            # same physical location (vertical line)
            # --------------------------------------------
            ax.vlines(
                x=space_prediction,
                ymin=reference_y,
                ymax=max(reference_y, comparison_y_at_space),
                color="black",
                linestyle="--",
                linewidth=1.5,
                zorder=2,
            )

    # --------------------------------------------------------
    # Horizontal ΔPSE
    # --------------------------------------------------------

    if (
        k_reference is not None
        or k_comparison is not None
    ):

        existing_k = [
            k
            for k in [
                k_reference,
                k_comparison,
            ]
            if k is not None
        ]

        y_max = (
            max(existing_k)
            * x_max
        )

        arrow_y = (
            y_max
            * 0.08
        )

        ax.annotate(
            "",
            xy=(
                pse,
                arrow_y,
            ),
            xytext=(
                reference_angle,
                arrow_y,
            ),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.4,
                "color": comparison_color,
            },
        )

        ax.text(
            (
                reference_angle
                + pse
            )
            / 2,
            arrow_y
            + y_max * 0.03,
            (
                f"ΔPSE = "
                f"{pse - reference_angle:+.1f}°"
            ),
            ha="center",
            fontsize=9,
        )

        ax.set_ylim(
            0,
            y_max * 1.05,
        )

    # --------------------------------------------------------
    # No previous k values
    # --------------------------------------------------------

    if not plotted_something:

        ax.text(
            0.5,
            0.5,
            "No previous k estimates\n"
            "for these frequencies",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="0.45",
        )

        ax.set_ylim(
            0,
            1,
        )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        x_max,
    )

    ax.set_xlabel(
        "Physical (°)"
    )

    ax.set_ylabel(
        "Perceived"
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    if plotted_something:

        ax.legend(
            frameon=False,
            fontsize=8,
        )

def _plot_prediction_panel(
    ax,
    fit_row,
    predictions,
):
    """
    Vertically show the observed PSE together with
    space- and cue-based predictions.
    """

    pse = float(
        fit_row["PSE"]
    )

    cue = str(
        fit_row["reference_cue"]
    ).upper()

    color = (
        COMPARISON_CUE_COLORS.get(
            cue,
            "C0",
        )
    )

    space_prediction = predictions[
        "space_prediction"
    ]

    cue_prediction = predictions[
        "cue_prediction"
    ]

    # --------------------------------------------------------
    # Space-based prediction
    #
    # Always BLACK DASHED
    # --------------------------------------------------------

    ax.axhline(
        space_prediction,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="Space-based",
    )

    # --------------------------------------------------------
    # Cue-based prediction
    #
    # Always BLACK SOLID
    # --------------------------------------------------------

    if cue_prediction is not None:

        ax.axhline(
            cue_prediction,
            color="black",
            linestyle="-",
            linewidth=1.6,
            label="Cue-based",
        )

    # --------------------------------------------------------
    # Observed PSE
    # --------------------------------------------------------

    ax.scatter(
        0.5,
        pse,
        s=90,
        color=color,
        zorder=4,
        label="Observed PSE",
        marker="s",
    )

    # --------------------------------------------------------
    # ΔPSE arrow:
    # space prediction -> observed PSE
    # --------------------------------------------------------

    ax.annotate(
        "",
        xy=(
            0.30,
            pse,
        ),
        xytext=(
            0.30,
            space_prediction,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.5,
            "color": color,
        },
    )

    delta_pse = (
        pse
        - space_prediction
    )

    ax.text(
        0.20,
        (
            pse
            + space_prediction
        )
        / 2,
        f"{delta_pse:+.1f}°",
        ha="right",
        va="center",
        fontsize=9,
        color=color,
    )

    # --------------------------------------------------------
    # Y range
    # --------------------------------------------------------

    values = [
        pse,
        space_prediction,
    ]

    if cue_prediction is not None:
        values.append(
            cue_prediction
        )

    spread = (
        max(values)
        - min(values)
    )

    margin = max(
        3,
        spread * 0.4,
    )

    ax.set_ylim(
        max(
            0,
            min(values) - margin,
        ),
        max(values) + margin,
    )

    # --------------------------------------------------------
    # Style
    # --------------------------------------------------------

    ax.set_xlim(
        0,
        1,
    )

    ax.set_xticks([])

    ax.set_ylabel(
        "Comparison location (°)"
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.spines[
        "bottom"
    ].set_visible(False)