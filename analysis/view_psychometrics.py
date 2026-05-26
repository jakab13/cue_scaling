# analysis/view_psychometrics.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def find_psychometric_summary(
    derivatives_root="data/psychometrics",
):
    summary_path = Path(derivatives_root) / "psychometric_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(
            f"No psychometric summary found at: {summary_path}"
        )

    return pd.read_csv(summary_path)


def filter_psychometric_summary(
    summary,
    subject_id=None,
    reference_cue=None,
    comparison_cue=None,
    reference_angle=None,
    reference_center_frequency=None,
    comparison_center_frequency=None,
    fit_name=None,
):
    """
    Filter psychometric summary table by condition parameters.
    """

    df = summary.copy()

    if subject_id is not None:
        df = df[df["subject_id"].astype(str) == str(subject_id)]

    if fit_name is not None:
        df = df[df["fit_name"].astype(str).str.contains(str(fit_name), regex=False)]

    if reference_cue is not None:
        df = df[df["reference_cue"].astype(str) == str(reference_cue)]

    if comparison_cue is not None:
        df = df[df["comparison_cue"].astype(str) == str(comparison_cue)]

    if reference_angle is not None:
        df = df[
            df["reference_angle_abs"].astype(float)
            == abs(float(reference_angle))
        ]

    if reference_center_frequency is not None:
        df = df[
            df["reference_center_frequency"].astype(float)
            == float(reference_center_frequency)
        ]

    if comparison_center_frequency is not None:
        df = df[
            df["comparison_center_frequency"].astype(float)
            == float(comparison_center_frequency)
        ]

    return df


def show_psychometric_figures(
    subject_id=None,
    derivatives_root="data/psychometrics",
    reference_cue=None,
    comparison_cue=None,
    reference_angle=None,
    reference_center_frequency=None,
    comparison_center_frequency=None,
    fit_name=None,
    max_figures=10,
):
    """
    Find and display existing psychometric fit figures.

    This does not fit anything. It only reads psychometric_summary.csv
    and opens the saved PNG figures.
    """

    summary = find_psychometric_summary(
        derivatives_root=derivatives_root,
    )

    matches = filter_psychometric_summary(
        summary=summary,
        subject_id=subject_id,
        reference_cue=reference_cue,
        comparison_cue=comparison_cue,
        reference_angle=reference_angle,
        reference_center_frequency=reference_center_frequency,
        comparison_center_frequency=comparison_center_frequency,
        fit_name=fit_name,
    )

    if matches.empty:
        print("No matching psychometric fits found.")
        return matches

    print("\nMatching psychometric fits:")
    print(
        matches[
            [
                "fit_name",
                "subject_id",
                "reference_cue",
                "reference_angle_abs",
                "reference_center_frequency",
                "comparison_cue",
                "comparison_center_frequency",
                "PSE_angle",
                "JND_84_angle",
                "PSE_value",
                "JND_84_value",
                "n_trials",
            ]
        ]
    )

    for _, row in matches.head(max_figures).iterrows():

        fig_path = Path(row["fit_figure"])

        if not fig_path.exists():
            print(f"Figure file not found: {fig_path}")
            continue

        img = mpimg.imread(fig_path)

        plt.figure(figsize=(7, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(row["fit_name"])
        plt.show()

    return matches