# analysis/estimate_k_slopes.py

from pathlib import Path
import numpy as np
import pandas as pd


def load_psychometric_summary(derivatives_root="data/psychometrics"):
    summary_path = Path(derivatives_root) / "psychometric_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"No summary table found at: {summary_path}")

    return pd.read_csv(summary_path)


def fit_no_intercept_slope(x, y):
    """
    Fit y = k * x.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)

    x = x[valid]
    y = y[valid]

    if len(x) == 0:
        return np.nan

    denominator = np.sum(x ** 2)

    if denominator == 0:
        return np.nan

    return np.sum(x * y) / denominator


def make_k_slope_points(
    summary,
    subject_id=None,
    reference_cues=("ILD", "ITD", "COMBINED"),
    comparison_cue="ILD",
    fit_domain="value",
):
    """
    Extract point-level PSE estimates used for k-slope analysis.

    One row = one psychometric PSE estimate.
    """

    df = summary.copy()

    if "fit_domain" not in df.columns:
        raise ValueError("psychometric_summary.csv must contain 'fit_domain'.")

    df = df[df["fit_domain"].astype(str).str.lower() == fit_domain.lower()]

    if subject_id is not None:
        df = df[df["subject_id"].astype(str) == str(subject_id)]

    df["reference_cue"] = df["reference_cue"].astype(str).str.upper()
    df["comparison_cue"] = df["comparison_cue"].astype(str).str.upper()

    df = df[df["reference_cue"].isin([cue.upper() for cue in reference_cues])]
    df = df[df["comparison_cue"] == comparison_cue.upper()]

    required_cols = [
        "reference_angle_folded",
        "reference_center_frequency",
        "comparison_center_frequency",
        "PSE",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["reference_angle_folded"] = pd.to_numeric(
        df["reference_angle_folded"], errors="coerce"
    )
    df["reference_center_frequency"] = pd.to_numeric(
        df["reference_center_frequency"], errors="coerce"
    )
    df["comparison_center_frequency"] = pd.to_numeric(
        df["comparison_center_frequency"], errors="coerce"
    )
    df["PSE"] = pd.to_numeric(df["PSE"], errors="coerce")

    # Exclude centered JND fits
    df = df[df["reference_angle_folded"] > 0]

    df = df.dropna(
        subset=[
            "reference_angle_folded",
            "reference_center_frequency",
            "comparison_center_frequency",
            "PSE",
        ]
    )

    keep_cols = [
        "subject_id",
        "reference_cue",
        "comparison_cue",
        "reference_center_frequency",
        "comparison_center_frequency",
        "reference_angle_folded",
        "PSE",
        "JND_84",
        "threshold_84",
        "x_unit",
        "fit_domain",
        "fit_name",
        "fit_json",
        "fit_figure",
        "n_trials",
        "n_levels",
        "source_files",
    ]

    keep_cols = [col for col in keep_cols if col in df.columns]

    return df[keep_cols].copy()


def add_sigma_y(slopes):
    """
    Derive single-cue sigma_y estimates from pairwise psychometric JNDs.

    The comparison cue is ILD.

    For ILD -> ILD:
        sigma_ILD = JND_ILD_ILD / sqrt(2)

    For cue -> ILD:
        sigma_cue = sqrt(JND_cue_ILD^2 - sigma_ILD^2)

    All sigma values are expressed in ILD comparison units (dB).
    """

    df = slopes.copy()

    df["reference_cue"] = (
        df["reference_cue"]
        .astype(str)
        .str.upper()
    )

    df["comparison_cue"] = (
        df["comparison_cue"]
        .astype(str)
        .str.upper()
    )

    df["JND_y_pair"] = pd.to_numeric(
        df["JND_y_pair"],
        errors="coerce",
    )

    # ------------------------------------------------------------
    # Estimate ILD comparison noise for each subject × frequency
    # ------------------------------------------------------------

    ild_rows = df[
        (df["reference_cue"] == "ILD")
        & (df["comparison_cue"] == "ILD")
        & (
            df["reference_center_frequency"]
            == df["comparison_center_frequency"]
        )
    ].copy()

    ild_rows["sigma_ILD_y"] = (
        ild_rows["JND_y_pair"] / np.sqrt(2)
    )

    # We only need one ILD noise estimate per subject and frequency.
    ild_baseline = (
        ild_rows[
            [
                "subject_id",
                "comparison_center_frequency",
                "sigma_ILD_y",
            ]
        ]
        .groupby(
            [
                "subject_id",
                "comparison_center_frequency",
            ],
            as_index=False,
        )
        ["sigma_ILD_y"]
        .median()
    )

    # ------------------------------------------------------------
    # Add ILD baseline to every cue at the same frequency
    # ------------------------------------------------------------

    df = df.merge(
        ild_baseline,
        on=[
            "subject_id",
            "comparison_center_frequency",
        ],
        how="left",
    )

    # ------------------------------------------------------------
    # Calculate reference-cue sigma
    # ------------------------------------------------------------

    pair_variance = df["JND_y_pair"] ** 2
    ild_variance = df["sigma_ILD_y"] ** 2

    reference_variance = pair_variance - ild_variance

    # ILD -> ILD is a special case:
    # both reference and comparison have the same sigma.
    is_ild = (
        (df["reference_cue"] == "ILD")
        & (df["comparison_cue"] == "ILD")
    )

    df["sigma_y"] = np.nan

    df.loc[is_ild, "sigma_y"] = df.loc[
        is_ild,
        "sigma_ILD_y",
    ]

    # ITD -> ILD and COMBINED -> ILD
    valid = (
        ~is_ild
        & np.isfinite(reference_variance)
        & (reference_variance >= 0)
    )

    df.loc[valid, "sigma_y"] = np.sqrt(
        reference_variance[valid]
    )

    # Useful diagnostic:
    # negative variance means the pairwise JND was smaller
    # than the independently estimated ILD noise.
    df["sigma_y_valid"] = (
        is_ild
        | valid
    )

    return df

def estimate_k_slope_summary(points):
    """
    Estimate no-intercept k slopes and cue-specific sigma_y values.

    One row = subject × frequency × reference cue × comparison cue.

    JND_y_pair:
        Median JND_84 across point estimates for that condition.
        This reflects uncertainty from both reference and comparison.

    sigma_y:
        Estimated uncertainty of the reference cue alone,
        expressed in comparison-cue (ILD) units.
    """

    group_cols = [
        "subject_id",
        "reference_cue",
        "comparison_cue",
        "reference_center_frequency",
        "comparison_center_frequency",
    ]

    rows = []

    for group_key, group_df in points.groupby(group_cols, dropna=False):

        group_info = dict(zip(group_cols, group_key))

        x = group_df["reference_angle_folded"].to_numpy(dtype=float)
        y = group_df["PSE"].to_numpy(dtype=float)

        jnd_values = pd.to_numeric(
            group_df["JND_84"],
            errors="coerce",
        )

        jnd_y_pair = jnd_values.median()

        k = fit_no_intercept_slope(x, y)

        row = {
            **group_info,
            "k_slope": k,

            # Psychometric width of the reference-comparison pair
            "JND_y_pair": jnd_y_pair,

            "n_points": len(group_df),

            "x_min": np.nanmin(x) if len(x) else np.nan,
            "x_max": np.nanmax(x) if len(x) else np.nan,
            "y_min": np.nanmin(y) if len(y) else np.nan,
            "y_max": np.nanmax(y) if len(y) else np.nan,

            "x_unit": "deg",
            "y_unit": (
                group_df["x_unit"].iloc[0]
                if "x_unit" in group_df.columns
                else ""
            ),
        }

        rows.append(row)

    slopes = pd.DataFrame(rows)

    # Convert pairwise JNDs into single-cue sigma estimates
    slopes = add_sigma_y(slopes)

    return slopes


def estimate_k_slopes(
    subject_id=None,
    derivatives_root="data/psychometrics",
    reference_cues=("ILD", "ITD", "COMBINED"),
    comparison_cue="ILD",
    fit_domain="value",
    save=True,
):
    """
    Create k-slope point and summary tables from psychometric fits.
    """

    derivatives_root = Path(derivatives_root)

    summary = load_psychometric_summary(
        derivatives_root=derivatives_root,
    )

    points = make_k_slope_points(
        summary=summary,
        subject_id=subject_id,
        reference_cues=reference_cues,
        comparison_cue=comparison_cue,
        fit_domain=fit_domain,
    )

    slopes = estimate_k_slope_summary(points)

    if save:
        points_path = derivatives_root / "k_slope_points.csv"
        slopes_path = derivatives_root / "k_slope_summary.csv"

        if points_path.exists():
            old_points = pd.read_csv(points_path)
            points_combined = pd.concat([old_points, points], ignore_index=True)

            points_combined = points_combined.drop_duplicates(
                subset=["subject_id", "fit_name"],
                keep="last",
            )
        else:
            points_combined = points

        if slopes_path.exists():
            old_slopes = pd.read_csv(slopes_path)
            slopes_combined = pd.concat([old_slopes, slopes], ignore_index=True)

            slopes_combined = slopes_combined.drop_duplicates(
                subset=[
                    "subject_id",
                    "reference_cue",
                    "comparison_cue",
                    "reference_center_frequency",
                    "comparison_center_frequency",
                ],
                keep="last",
            )
        else:
            slopes_combined = slopes

        points_combined.to_csv(points_path, index=False)
        slopes_combined.to_csv(slopes_path, index=False)

        print(f"Saved k-slope points to: {points_path}")
        print(f"Saved k-slope summary to: {slopes_path}")

    return points, slopes