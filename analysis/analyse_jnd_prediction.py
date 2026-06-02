# analysis/analyse_jnd_prediction.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CUE_ORDER = ["ILD", "ITD", "COMBINED"]

CUE_COLORS = {
    "ILD": "C0",
    "ITD": "C1",
    "COMBINED": "C2",
}


def load_psychometric_summary(derivatives_root="data/psychometrics"):
    summary_path = Path(derivatives_root) / "psychometric_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"No summary table found at: {summary_path}")

    return pd.read_csv(summary_path)


def load_k_slope_summary(derivatives_root="data/psychometrics"):
    slopes_path = Path(derivatives_root) / "k_slope_summary.csv"

    if not slopes_path.exists():
        raise FileNotFoundError(
            f"No k-slope summary found at: {slopes_path}. "
            "Run estimate_k_slopes(...) first."
        )

    return pd.read_csv(slopes_path)


def extract_measured_jnds(
    psychometric_summary,
    subject_id=None,
    cues=("ILD", "ITD", "COMBINED"),
    fit_domain="angle",
):
    """
    Extract centred within-cue JNDs from psychometric summary.

    These are the measured JNDs from the JND task:
        reference_angle_folded == 0
        reference_cue == comparison_cue
        fit_domain == angle

    JND_84 is in degrees.
    """

    df = psychometric_summary.copy()

    df = df[df["fit_domain"].astype(str).str.lower() == fit_domain.lower()]

    if subject_id is not None:
        df = df[df["subject_id"].astype(str) == str(subject_id)]

    df["reference_cue"] = df["reference_cue"].astype(str).str.upper()
    df["comparison_cue"] = df["comparison_cue"].astype(str).str.upper()

    df = df[df["reference_cue"] == df["comparison_cue"]]
    df = df[df["reference_cue"].isin([cue.upper() for cue in cues])]

    df["reference_angle_folded"] = pd.to_numeric(
        df["reference_angle_folded"],
        errors="coerce",
    )

    df = df[df["reference_angle_folded"] == 0]

    df["reference_center_frequency"] = pd.to_numeric(
        df["reference_center_frequency"],
        errors="coerce",
    )

    df["JND_84"] = pd.to_numeric(df["JND_84"], errors="coerce")

    df = df.dropna(
        subset=[
            "subject_id",
            "reference_cue",
            "reference_center_frequency",
            "JND_84",
        ]
    )

    out = df[
        [
            "subject_id",
            "reference_cue",
            "reference_center_frequency",
            "JND_84",
            "fit_name",
            "fit_json",
            "fit_figure",
        ]
    ].copy()

    out = out.rename(
        columns={
            "reference_center_frequency": "frequency",
            "JND_84": "JND_measured",
            "fit_name": "JND_fit_name",
            "fit_json": "JND_fit_json",
            "fit_figure": "JND_fit_figure",
        }
    )

    return out


def prepare_k_slope_predictors(
    k_slope_summary,
    subject_id=None,
    cues=("ILD", "ITD", "COMBINED"),
    comparison_cue="ILD",
):
    """
    Prepare k and sigma_y predictors from k_slope_summary.

    Assumes k_slope_summary contains:
        k_slope
        sigma_y

    where:
        k_slope is in dB/degree
        sigma_y is in dB
    """

    df = k_slope_summary.copy()

    if subject_id is not None:
        df = df[df["subject_id"].astype(str) == str(subject_id)]

    df["reference_cue"] = df["reference_cue"].astype(str).str.upper()
    df["comparison_cue"] = df["comparison_cue"].astype(str).str.upper()

    df = df[df["reference_cue"].isin([cue.upper() for cue in cues])]
    df = df[df["comparison_cue"] == comparison_cue.upper()]

    df["reference_center_frequency"] = pd.to_numeric(
        df["reference_center_frequency"],
        errors="coerce",
    )
    df["k_slope"] = pd.to_numeric(df["k_slope"], errors="coerce")
    df["sigma_y"] = pd.to_numeric(df["sigma_y"], errors="coerce")

    df = df.dropna(
        subset=[
            "subject_id",
            "reference_cue",
            "reference_center_frequency",
            "k_slope",
            "sigma_y",
        ]
    )

    out = df[
        [
            "subject_id",
            "reference_cue",
            "reference_center_frequency",
            "comparison_cue",
            "comparison_center_frequency",
            "k_slope",
            "sigma_y",
            "n_points",
        ]
    ].copy()

    out = out.rename(
        columns={
            "reference_center_frequency": "frequency",
            "n_points": "n_k_points",
        }
    )

    return out


def make_jnd_prediction_table(
    subject_id=None,
    derivatives_root="data/psychometrics",
    cues=("ILD", "ITD", "COMBINED"),
    comparison_cue="ILD",
    save=True,
):
    """
    Build table comparing measured JNDs against sigma_y / k predictions.
    """

    derivatives_root = Path(derivatives_root)

    psychometric_summary = load_psychometric_summary(derivatives_root)
    k_slope_summary = load_k_slope_summary(derivatives_root)

    measured_jnds = extract_measured_jnds(
        psychometric_summary=psychometric_summary,
        subject_id=subject_id,
        cues=cues,
        fit_domain="angle",
    )

    predictors = prepare_k_slope_predictors(
        k_slope_summary=k_slope_summary,
        subject_id=subject_id,
        cues=cues,
        comparison_cue=comparison_cue,
    )

    table = measured_jnds.merge(
        predictors,
        on=["subject_id", "reference_cue", "frequency"],
        how="inner",
    )

    table["JND_predicted"] = table["sigma_y"] / table["k_slope"]

    table["prediction_error"] = (
        table["JND_measured"] - table["JND_predicted"]
    )

    table["prediction_ratio"] = (
        table["JND_measured"] / table["JND_predicted"]
    )

    table["abs_prediction_error"] = table["prediction_error"].abs()

    # Avoid impossible/infinite rows
    table = table.replace([np.inf, -np.inf], np.nan)
    table = table.dropna(
        subset=[
            "JND_measured",
            "JND_predicted",
            "k_slope",
            "sigma_y",
        ]
    )

    if save:
        out_path = derivatives_root / "jnd_prediction_summary.csv"

        if out_path.exists():
            old = pd.read_csv(out_path)
            combined = pd.concat([old, table], ignore_index=True)

            combined = combined.drop_duplicates(
                subset=[
                    "subject_id",
                    "reference_cue",
                    "frequency",
                ],
                keep="last",
            )
        else:
            combined = table

        combined.to_csv(out_path, index=False)
        print(f"Saved JND prediction summary to: {out_path}")

    return table


def plot_jnd_prediction(
    subject_id=None,
    derivatives_root="data/psychometrics",
    frequencies=None,
    cues=("ILD", "ITD", "COMBINED"),
    save=False,
    save_path=None,
):
    """
    Plot measured JND against predicted JND = sigma_y / k.
    """

    derivatives_root = Path(derivatives_root)

    prediction_path = derivatives_root / "jnd_prediction_summary.csv"

    if not prediction_path.exists():
        table = make_jnd_prediction_table(
            subject_id=subject_id,
            derivatives_root=derivatives_root,
            cues=cues,
            save=True,
        )
    else:
        table = pd.read_csv(prediction_path)

    if subject_id is not None:
        table = table[table["subject_id"].astype(str) == str(subject_id)]

    table["reference_cue"] = table["reference_cue"].astype(str).str.upper()

    table = table[table["reference_cue"].isin([cue.upper() for cue in cues])]

    if frequencies is not None:
        table = table[
            table["frequency"].astype(float).isin([float(f) for f in frequencies])
        ]

    if table.empty:
        raise ValueError("No JND prediction data found.")

    fig, ax = plt.subplots(figsize=(5, 5))

    for cue in CUE_ORDER:
        cue_df = table[table["reference_cue"] == cue]

        if cue_df.empty:
            continue

        color = CUE_COLORS.get(cue, None)

        ax.scatter(
            cue_df["JND_predicted"],
            cue_df["JND_measured"],
            label=cue,
            color=color,
            alpha=0.9,
        )

    max_val = np.nanmax(
        [
            table["JND_predicted"].max(),
            table["JND_measured"].max(),
        ]
    )

    ax.plot(
        [0, max_val],
        [0, max_val],
        color="lightgray",
        linewidth=1.5,
        zorder=0,
    )

    ax.set_xlim(0, max_val * 1.05)
    ax.set_ylim(0, max_val * 1.05)

    ax.set_xlabel("Predicted JND = sigma / k (deg)")
    ax.set_ylabel("Measured JND (deg)")
    ax.set_title("Measured vs predicted JND")

    ax.legend(title="Cue")

    fig.tight_layout()

    if save:
        if save_path is None:
            save_dir = derivatives_root / "figures"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "jnd_prediction_measured_vs_predicted.png"

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved JND prediction plot to: {save_path}")

    return fig, ax, table