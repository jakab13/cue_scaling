# experiment/experiment_planner.py

from pathlib import Path
import numpy as np
import pandas as pd
from stimuli.sound_handler import ild_slope_at_zero_fit


# ============================================================
# Defaults
# ============================================================

DEFAULT_ILS_PATH = "stimuli/ils_kemar.pickle"

DEFAULT_COMPARISON_VALUE_OFFSETS = [
    -7.0, -5.0, -3.0, -1.0,
     1.0, 3.0, 5.0, 7.0
]

DEFAULT_REFERENCE_ANGLE_SETS = [
    [1, 2, 3],
    [2, 4, 6],
    [3, 6, 9],
    [4, 8, 12],
    [5, 10, 15],
    [6, 12, 18],
    [8, 16, 24],
]

DEFAULT_VALUE_LIMIT = 12
DEFAULT_ANGLE_LIMIT = 30


# Approximate fallback k estimates in dB/degree.
# These are only used when no subject-specific or manually supplied estimate exists.
# Values are deliberately rough and conservative.
DEFAULT_K_ESTIMATES = {
    "ILD": {
        400: 0.09,
        600: 0.15,
        800: 0.225,
        1000: 0.25,
        1200: 0.205,
        1400: 0.14,
        1600: 0.14,
        1800: 0.14,
        2000: 0.14,
        2200: 0.14,
        2400: 0.14,
    },
    "ITD": {
        400: 0.16,
        600: 0.245,
        800: 0.185,
        1000: 0.13,
        1200: 0.105,
        1400: 0.065,
        1600: 0.065,
        1800: 0.065,
        2000: 0.065,
        2200: 0.065,
        2400: 0.065,
    },
    "COMBINED": {
        400: 0.40,
        600: 0.51,
        800: 0.515,
        1000: 0.48,
        1200: 0.385,
        1400: 0.25,
        1600: 0.25,
        1800: 0.25,
        2000: 0.25,
        2200: 0.25,
        2400: 0.25,
    },
}


DEFAULT_JND_COMPARISON_ANGLES = {
    "ILD": [-35, -25, -15, -5, 5, 15, 25, 35],
    "ITD": [-35, -25, -15, -5, 5, 15, 25, 35],
    "COMBINED": [-21, -15, -9, -3, 3, 9, 15, 21],
}


def load_k_slopes(derivatives_root="data/psychometrics"):
    """
    Load previous k-slope estimates if available.
    """

    path = Path(derivatives_root) / "k_slope_summary.csv"

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def nearest_frequency(frequency, available_frequencies):
    """
    Find nearest frequency in a list/dict of available frequencies.
    """

    available = np.asarray(list(available_frequencies), dtype=float)

    if len(available) == 0:
        return None

    idx = np.argmin(np.abs(available - float(frequency)))

    return int(available[idx])


def get_k_estimate(
    reference_cue,
    frequency,
    subject_id=None,
    comparison_cue="ILD",
    k_estimate=None,
    derivatives_root="data/psychometrics",
):
    """
    Get the best available k estimate in dB/degree.

    Priority:
        1. manually supplied k_estimate
        2. previous subject-specific estimate
        3. previous pooled median estimate
        4. built-in fallback estimate
    """

    reference_cue = reference_cue.upper()
    comparison_cue = comparison_cue.upper()
    frequency = float(frequency)

    # # ------------------------------------------------------------
    # # 1. Manual estimate
    # # ------------------------------------------------------------
    # if k_estimate is not None:
    #     return float(k_estimate), "manual"
    #
    # # ------------------------------------------------------------
    # # 2–3. Previous estimates
    # # ------------------------------------------------------------
    # slopes = load_k_slopes(derivatives_root)
    #
    # if not slopes.empty:
    #     slopes = slopes.copy()
    #
    #     slopes["reference_cue"] = slopes["reference_cue"].astype(str).str.upper()
    #     slopes["comparison_cue"] = slopes["comparison_cue"].astype(str).str.upper()
    #     slopes["reference_center_frequency"] = pd.to_numeric(
    #         slopes["reference_center_frequency"],
    #         errors="coerce",
    #     )
    #     slopes["k_slope"] = pd.to_numeric(
    #         slopes["k_slope"],
    #         errors="coerce",
    #     )
    #
    #     df = slopes[
    #         (slopes["reference_cue"] == reference_cue)
    #         & (slopes["comparison_cue"] == comparison_cue)
    #         & (slopes["reference_center_frequency"] == frequency)
    #     ].copy()
    #
    #     # 2. Subject-specific estimate
    #     if subject_id is not None and not df.empty:
    #         subject_df = df[df["subject_id"].astype(str) == str(subject_id)]
    #
    #         if not subject_df.empty:
    #             k = subject_df["k_slope"].dropna().iloc[-1]
    #             return float(k), "previous subject estimate"
    #
    #     # 3. Pooled median estimate
    #     if not df.empty:
    #         k = df["k_slope"].dropna().median()
    #
    #         if np.isfinite(k):
    #             return float(k), "previous pooled estimate"
    #
    # # ------------------------------------------------------------
    # # 4. ILD fallback from actual ILS file, if available
    # # ------------------------------------------------------------
    # if reference_cue == "ILD":
    #     try:
    #         k = ild_slope_at_zero_fit(
    #             freq_hz=frequency,
    #         )
    #         return float(k), "ILD slope from ILS"
    #     except Exception as e:
    #         print(
    #             f"Could not estimate ILD slope from ILS: {e}. "
    #             "Using built-in fallback estimate instead."
    #         )

    # ------------------------------------------------------------
    # 5. Built-in fallback estimate
    # ------------------------------------------------------------
    if reference_cue in DEFAULT_K_ESTIMATES:
        freq_dict = DEFAULT_K_ESTIMATES[reference_cue]
        nearest_freq = nearest_frequency(frequency, freq_dict.keys())

        if nearest_freq is not None:
            return float(freq_dict[nearest_freq]), f"fallback estimate ({nearest_freq} Hz)"

    raise ValueError(
        f"No k estimate available for {reference_cue} at {frequency} Hz."
    )


def check_angle_set(
    reference_angles,
    k_slope,
    comparison_value_offsets=DEFAULT_COMPARISON_VALUE_OFFSETS,
    value_limit=DEFAULT_VALUE_LIMIT,
    angle_limit=DEFAULT_ANGLE_LIMIT,
):
    """
    Check whether a candidate reference-angle set is safe.

    A set is valid if:
        reference angles stay within angle_limit
        all comparison values stay within ±value_limit
    """

    reference_angles = np.asarray(reference_angles, dtype=float)
    offsets = np.asarray(comparison_value_offsets, dtype=float)

    pse_values = k_slope * reference_angles

    min_values = pse_values + offsets.min()
    max_values = pse_values + offsets.max()

    valid = (
        np.all(reference_angles <= angle_limit)
        and np.all(min_values >= -value_limit)
        and np.all(max_values <= value_limit)
    )

    return {
        "reference_angles": reference_angles.tolist(),
        "pse_estimate_values": pse_values.tolist(),
        "min_comparison_values": min_values.tolist(),
        "max_comparison_values": max_values.tolist(),
        "valid": bool(valid),
        "max_reference_angle": float(reference_angles.max()),
        "min_value": float(np.min(min_values)),
        "max_value": float(np.max(max_values)),
    }


def choose_angle_set(
    k_slope,
    candidate_angle_sets=DEFAULT_REFERENCE_ANGLE_SETS,
    comparison_value_offsets=DEFAULT_COMPARISON_VALUE_OFFSETS,
    value_limit=DEFAULT_VALUE_LIMIT,
    angle_limit=DEFAULT_ANGLE_LIMIT,
):
    """
    Choose the widest valid reference-angle set.
    """

    checks = []

    for angle_set in candidate_angle_sets:
        check = check_angle_set(
            reference_angles=angle_set,
            k_slope=k_slope,
            comparison_value_offsets=comparison_value_offsets,
            value_limit=value_limit,
            angle_limit=angle_limit,
        )

        checks.append(check)

    valid_sets = [check for check in checks if check["valid"]]

    if not valid_sets:
        return None, checks

    best = max(valid_sets, key=lambda check: check["max_reference_angle"])

    return best, checks


def print_k_plan(plan):
    """
    Print a compact planning summary.
    """

    chosen = plan["chosen"]

    print("\nK-slope plan")
    print("-" * 50)
    print(f"Reference cue: {plan['reference_cue']}")
    print(f"Comparison cue:{plan['comparison_cue']}")
    print(f"Frequency:     {plan['frequency']} Hz")
    print(f"k estimate:    {plan['k_estimate']:.3f} dB/deg")
    print(f"k source:      {plan['k_source']}")
    print()
    print(f"Angles:        {chosen['reference_angles']}")
    print(
        "PSE values:    "
        f"{[round(v, 3) for v in chosen['pse_estimate_values']]}"
    )
    print(
        "Value range:   "
        f"{round(chosen['min_value'], 3)} to {round(chosen['max_value'], 3)} dB"
    )
    print("-" * 50)


def plan_k_slope_run(
    reference_cue,
    frequency,
    subject_id=None,
    comparison_cue="ILD",
    k_estimate=None,
    comparison_value_offsets=DEFAULT_COMPARISON_VALUE_OFFSETS,
    candidate_angle_sets=DEFAULT_REFERENCE_ANGLE_SETS,
    value_limit=DEFAULT_VALUE_LIMIT,
    angle_limit=DEFAULT_ANGLE_LIMIT,
    derivatives_root="data/psychometrics",
    verbose=False,
):
    """
    Plan a k-slope run.

    Returns
    -------
    reference_points : list of dict
        Can be passed directly to make_k_slope_run().

    plan : dict
        Details about the selected plan.
    """

    reference_cue = reference_cue.upper()
    comparison_cue = comparison_cue.upper()

    k, source = get_k_estimate(
        reference_cue=reference_cue,
        frequency=frequency,
        subject_id=subject_id,
        comparison_cue=comparison_cue,
        k_estimate=k_estimate,
        derivatives_root=derivatives_root,
    )

    best, checks = choose_angle_set(
        k_slope=k,
        candidate_angle_sets=candidate_angle_sets,
        comparison_value_offsets=comparison_value_offsets,
        value_limit=value_limit,
        angle_limit=angle_limit,
    )

    if best is None:
        raise ValueError(
            f"No safe angle set found for {reference_cue} at {frequency} Hz "
            f"with k={k:.3f} dB/deg. Try smaller offsets or smaller angles."
        )

    reference_points = []

    for angle, pse_value in zip(
        best["reference_angles"],
        best["pse_estimate_values"],
    ):
        reference_points.append(
            {
                "reference_cue": reference_cue,
                "reference_angle": float(angle),
                "pse_estimate_value": float(pse_value),
            }
        )

    plan = {
        "reference_cue": reference_cue,
        "comparison_cue": comparison_cue,
        "frequency": frequency,
        "k_estimate": k,
        "k_source": source,
        "chosen": best,
        "all_checks": checks,
        "comparison_value_offsets": comparison_value_offsets,
        "value_limit": value_limit,
        "angle_limit": angle_limit,
    }

    if verbose:
        print_k_plan(plan)

    return reference_points


def plan_jnd_run(
    cue,
    frequency=None,
    comparison_angles=None,
    verbose=False,
):
    """
    Plan a centred JND run.

    The reference is always 0 degrees.
    The comparison angles are selected from simple cue-specific defaults,
    unless comparison_angles are provided manually.

    Returns
    -------
    comparison_angles : list
        Angles to pass to make_jnd_run().
    plan : dict
        Small summary of the selected settings.
    """

    cue = cue.upper()

    if comparison_angles is None:
        if cue not in DEFAULT_JND_COMPARISON_ANGLES:
            raise ValueError(
                f"No default JND comparison angles defined for cue '{cue}'. "
                "Use ILD, ITD, COMBINED, or provide comparison_angles manually."
            )

        comparison_angles = DEFAULT_JND_COMPARISON_ANGLES[cue]

    comparison_angles = [float(angle) for angle in comparison_angles]

    plan = {
        "cue": cue,
        "frequency": frequency,
        "reference_angle": 0,
        "comparison_angles": comparison_angles,
        "min_angle": min(comparison_angles),
        "max_angle": max(comparison_angles),
        "n_levels": len(comparison_angles),
    }

    if verbose:
        print_jnd_plan(plan)

    return comparison_angles


def print_jnd_plan(plan):
    """
    Print compact JND planning summary.
    """

    print("\nJND plan")
    print("-" * 50)
    print(f"Cue:             {plan['cue']}")
    if plan["frequency"] is not None:
        print(f"Frequency:       {plan['frequency']} Hz")
    print(f"Reference angle: {plan['reference_angle']}°")
    print(f"Comparisons:     {plan['comparison_angles']}")
    print(
        f"Range:           {plan['min_angle']}° to {plan['max_angle']}° "
        f"({plan['n_levels']} levels)"
    )
    print("-" * 50)