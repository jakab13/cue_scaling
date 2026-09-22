from pathlib import Path

import numpy as np
import pandas as pd

from stimuli.sound_handler import (
    get_interaural_level_spectrum,
    ild_from_ils,
)


# ============================================================
# SETTINGS
# ============================================================

REFERENCE_ANGLE = 12.0

# Candidate frequency regimes
LOW_ILD_FREQUENCIES = [
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    1600,
    1700,
    1800,
]

HIGH_ILD_FREQUENCIES = [
    2000,
    2200,
    2400,
    2600,
    2800,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    12000,
    14000,
    16000,
    18000
]

COMBINED_FREQUENCIES = [
    600,
    800,
    1000,
    1200,
    1400,
    1600,
]

# How different may the frequencies be?
MAX_OCTAVE_DISTANCE = 3.0

# We want at least this much separation between
# the space-based and cue-based prediction.
MIN_MODEL_SEPARATION_DEG = 5.0

# Keep predicted PSEs within sensible spatial bounds.
MIN_PSE_ANGLE = 2.0
MAX_PSE_ANGLE = 30.0

# Comparison sampling
N_COMPARISON_ANGLES = 9
COMPARISON_MARGIN_DEG = 4.0

# Empirical combined-cue slopes
K_SLOPE_PATH = Path(
    "data/psychometrics/k_slope_summary.csv"
)

SAVE_DIR = Path(
    "stimuli/frequency_alignment_planning"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def octave_distance(f1, f2):
    """
    Absolute frequency distance in octaves.
    """
    return abs(
        np.log2(float(f2) / float(f1))
    )


def make_comparison_angles(
    space_prediction,
    cue_prediction,
    n_angles=N_COMPARISON_ANGLES,
    margin=COMPARISON_MARGIN_DEG,
    min_angle=0,
    max_angle=MAX_PSE_ANGLE,
):
    """
    Generate a simple comparison-angle range that comfortably
    contains both theoretical predictions.
    """

    low = min(
        space_prediction,
        cue_prediction,
    ) - margin

    high = max(
        space_prediction,
        cue_prediction,
    ) + margin

    low = max(
        min_angle,
        low,
    )

    high = min(
        max_angle,
        high,
    )

    angles = np.linspace(
        low,
        high,
        n_angles,
    )

    # Round to convenient whole-degree values
    angles = np.round(
        angles
    ).astype(int)

    # Remove accidental duplicates after rounding
    angles = np.unique(
        angles
    )

    return angles.tolist()


# ============================================================
# ILD PREDICTIONS
# ============================================================

def ild_magnitude(
    angle,
    frequency,
    ils,
):
    """
    Absolute ILD magnitude at an azimuth/frequency.
    """

    ild = ild_from_ils(
        azis_deg=float(angle),
        freq_hz=float(frequency),
        ils_dict=ils,
    )

    return abs(float(ild))


def angle_for_ild(
    target_ild,
    frequency,
    ils,
    angle_min=0,
    angle_max=MAX_PSE_ANGLE,
    step=0.1,
):
    """
    Find the comparison-frequency azimuth that gives
    the ILD closest to target_ild.
    """

    angles = np.arange(
        angle_min,
        angle_max + step,
        step,
    )

    ild_values = np.abs(
        ild_from_ils(
            azis_deg=angles,
            freq_hz=frequency,
            ils_dict=ils,
        )
    )

    idx = np.argmin(
        np.abs(
            ild_values - target_ild
        )
    )

    best_angle = float(
        angles[idx]
    )

    best_ild = float(
        ild_values[idx]
    )

    error_db = abs(
        best_ild - target_ild
    )

    return best_angle, error_db


def get_ild_pair_candidates(
    frequencies,
    reference_angle=REFERENCE_ANGLE,
    ils=None,
):
    """
    Evaluate all ordered ILD frequency pairs.

    Space prediction:
        comparison angle = reference angle

    Cue prediction:
        ILD_comparison(angle) = ILD_reference(reference_angle)
    """

    if ils is None:
        ils = get_interaural_level_spectrum()

    rows = []

    for ref_freq in frequencies:

        ref_ild = ild_magnitude(
            angle=reference_angle,
            frequency=ref_freq,
            ils=ils,
        )

        for comp_freq in frequencies:

            if comp_freq == ref_freq:
                continue

            freq_distance = octave_distance(
                ref_freq,
                comp_freq,
            )

            if freq_distance > MAX_OCTAVE_DISTANCE:
                continue

            cue_angle, ild_error = angle_for_ild(
                target_ild=ref_ild,
                frequency=comp_freq,
                ils=ils,
            )

            space_angle = float(
                reference_angle
            )

            separation = abs(
                cue_angle - space_angle
            )

            valid = (
                MIN_PSE_ANGLE
                <= cue_angle
                <= MAX_PSE_ANGLE
            )

            comparison_angles = (
                make_comparison_angles(
                    space_prediction=space_angle,
                    cue_prediction=cue_angle,
                )
                if valid
                else []
            )

            rows.append(
                {
                    "reference_frequency": ref_freq,
                    "comparison_frequency": comp_freq,

                    "reference_angle": reference_angle,

                    "reference_ild_db": ref_ild,

                    "space_prediction_deg": space_angle,
                    "cue_prediction_deg": cue_angle,

                    "model_separation_deg": separation,

                    "octave_distance": freq_distance,

                    "ild_match_error_db": ild_error,

                    "comparison_angles": comparison_angles,

                    "valid": valid,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# COMBINED-CUE PREDICTIONS
# ============================================================

def load_combined_k_slopes(
    path=K_SLOPE_PATH,
):
    """
    Load group-median COMBINED k slopes from the current
    psychometric k-slope summary.

    Returns one k value per frequency.
    """

    df = pd.read_csv(
        path
    )

    df = df.copy()

    df["reference_cue"] = (
        df["reference_cue"]
        .astype(str)
        .str.upper()
    )

    if "comparison_cue" in df.columns:
        df["comparison_cue"] = (
            df["comparison_cue"]
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

    combined = df[
        df["reference_cue"]
        == "COMBINED"
    ].copy()

    # Current k-slope experiment uses ILD comparison space.
    if "comparison_cue" in combined.columns:
        combined = combined[
            combined["comparison_cue"]
            == "ILD"
        ]

    summary = (
        combined
        .groupby(
            "reference_center_frequency",
            as_index=False,
        )
        ["k_slope"]
        .median()
    )

    summary = summary.rename(
        columns={
            "reference_center_frequency":
                "frequency",
            "k_slope":
                "combined_k",
        }
    )

    return summary


def get_combined_pair_candidates(
    frequencies=COMBINED_FREQUENCIES,
    reference_angle=REFERENCE_ANGLE,
    slope_path=K_SLOPE_PATH,
):
    """
    Evaluate ordered COMBINED frequency pairs.

    Space prediction:
        comparison angle = reference angle

    Scaling prediction:
        k_ref * angle_ref = k_comp * angle_comp

    therefore:
        angle_comp = (k_ref / k_comp) * angle_ref
    """

    slopes = load_combined_k_slopes(
        path=slope_path,
    )

    k_lookup = dict(
        zip(
            slopes["frequency"],
            slopes["combined_k"],
        )
    )

    available_frequencies = [
        f
        for f in frequencies
        if f in k_lookup
    ]

    rows = []

    for ref_freq in available_frequencies:

        for comp_freq in available_frequencies:

            if ref_freq == comp_freq:
                continue

            freq_distance = octave_distance(
                ref_freq,
                comp_freq,
            )

            if freq_distance > MAX_OCTAVE_DISTANCE:
                continue

            k_ref = float(
                k_lookup[ref_freq]
            )

            k_comp = float(
                k_lookup[comp_freq]
            )

            if (
                not np.isfinite(k_ref)
                or not np.isfinite(k_comp)
                or k_comp <= 0
            ):
                continue

            space_angle = float(
                reference_angle
            )

            scaling_angle = (
                k_ref
                / k_comp
                * reference_angle
            )

            separation = abs(
                scaling_angle
                - space_angle
            )

            valid = (
                MIN_PSE_ANGLE
                <= scaling_angle
                <= MAX_PSE_ANGLE
            )

            comparison_angles = (
                make_comparison_angles(
                    space_prediction=space_angle,
                    cue_prediction=scaling_angle,
                )
                if valid
                else []
            )

            rows.append(
                {
                    "reference_frequency": ref_freq,
                    "comparison_frequency": comp_freq,

                    "reference_angle": reference_angle,

                    "k_reference": k_ref,
                    "k_comparison": k_comp,

                    "space_prediction_deg": space_angle,
                    "scaling_prediction_deg": scaling_angle,

                    "model_separation_deg": separation,

                    "octave_distance": freq_distance,

                    "comparison_angles": comparison_angles,

                    "valid": valid,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# RANKING
# ============================================================

def rank_candidates(
    df,
    min_separation=MIN_MODEL_SEPARATION_DEG,
):
    """
    Keep useful pairs, then prefer:

        1. smallest frequency distance
        2. largest model separation

    This directly implements:
        'as spectrally close as possible,
         while still giving useful model separation'.
    """

    ranked = df[
        df["valid"]
        & (
            df["model_separation_deg"]
            >= min_separation
        )
    ].copy()

    ranked = ranked.sort_values(
        by=[
            "octave_distance",
            "model_separation_deg",
        ],
        ascending=[
            True,
            False,
        ],
    )

    return ranked.reset_index(
        drop=True
    )


# ============================================================
# PRINTING
# ============================================================

def print_top(
    title,
    df,
    n=8,
):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    if df.empty:
        print(
            "No candidates passed the current criteria."
        )
        return

    columns = [
        "reference_frequency",
        "comparison_frequency",
        "space_prediction_deg",
    ]

    if "cue_prediction_deg" in df.columns:
        columns.append(
            "cue_prediction_deg"
        )

    if "scaling_prediction_deg" in df.columns:
        columns.append(
            "scaling_prediction_deg"
        )

    columns += [
        "model_separation_deg",
        "octave_distance",
        "comparison_angles",
    ]

    print(
        df[columns]
        .head(n)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def plan_frequency_pairs():

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ils = get_interaural_level_spectrum()

    # --------------------------------------------------------
    # Block 1: high-frequency ILD
    # --------------------------------------------------------

    high_ild = get_ild_pair_candidates(
        frequencies=HIGH_ILD_FREQUENCIES,
        ils=ils,
    )

    high_ild = rank_candidates(
        high_ild
    )

    # --------------------------------------------------------
    # Block 2: low-frequency ILD
    # --------------------------------------------------------

    low_ild = get_ild_pair_candidates(
        frequencies=LOW_ILD_FREQUENCIES,
        ils=ils,
    )

    low_ild = rank_candidates(
        low_ild
    )

    # --------------------------------------------------------
    # Block 3: combined cues
    # --------------------------------------------------------

    combined = get_combined_pair_candidates()

    combined = rank_candidates(
        combined
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_top(
        "BLOCK 1: HIGH-FREQUENCY ILD -> ILD",
        high_ild,
    )

    print_top(
        "BLOCK 2: LOW-FREQUENCY ILD -> ILD",
        low_ild,
    )

    print_top(
        "BLOCK 3: COMBINED -> COMBINED",
        combined,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    high_ild.to_csv(
        SAVE_DIR
        / "high_frequency_ild_candidates.csv",
        index=False,
    )

    low_ild.to_csv(
        SAVE_DIR
        / "low_frequency_ild_candidates.csv",
        index=False,
    )

    combined.to_csv(
        SAVE_DIR
        / "combined_candidates.csv",
        index=False,
    )

    return {
        "high_ild": high_ild,
        "low_ild": low_ild,
        "combined": combined,
    }


if __name__ == "__main__":
    tables = plan_frequency_pairs()