# experiment/experiment_pipeline.py

from pathlib import Path
import pandas as pd

from experiment.trial_sequence import build_run_sequence
from experiment.run_experiment import run_experiment
from analysis.fit_psychometrics import fit_run_file, fit_run_files


def run_experiment_from_conditions(
    subject_id,
    conditions,
    save_root="data/raw",
    randomize=True,
    seed=None,
    left_key="1",
    right_key="2",
    analysis_role="main",
    include_in_analysis=True,
):
    """
    Build a trial sequence from ConditionSpec objects and run the experiment.

    This function only runs the experiment and saves one raw CSV file.
    It does not fit or plot anything.

    Parameters
    ----------
    subject_id : str
        Participant identifier.

    conditions : list
        List of ConditionSpec objects.

    save_root : str
        Root directory for raw data.

    randomize : bool
        Whether to randomize trial order.

    seed : int or None
        Random seed for trial randomization.

    left_key, right_key : str
        Response keys.

    Returns
    -------
    csv_path : Path or None
        Path to the saved raw data file, or None if the run was cancelled.
    """

    run_trials = build_run_sequence(
        conditions=conditions,
        randomize=randomize,
        seed=seed,
    )

    print("\nPrepared run")
    print("-" * 50)
    print(f"Subject ID: {subject_id}")
    print(f"Conditions: {[c.condition_id for c in conditions]}")
    print(f"Number of trials: {len(run_trials)}")
    print("-" * 50)

    if not include_in_analysis:
        print("This run is marked as practice/test and will not enter main analysis.")

    csv_path = run_experiment(
        subject_id=subject_id,
        run_trials=run_trials,
        save_root=save_root,
        left_key=left_key,
        right_key=right_key,
        analysis_role=analysis_role,
        include_in_analysis=include_in_analysis,
    )

    return csv_path


def find_run_files(
    subject_id,
    save_root="data/raw",
    condition_id=None,
    reference_cue=None,
    comparison_cue=None,
    reference_angle=None,
    reference_center_frequency=None,
    comparison_center_frequency=None,
):
    """
    Find raw run CSV files matching a set of experiment parameters.

    The search is based on metadata inside the CSV files, not on filenames.

    Parameters
    ----------
    subject_id : str
        Participant identifier.

    save_root : str
        Root directory for raw data.

    condition_id : str or None
        Optional condition label. Usually leave this as None if you want to
        pool across repeated runs or renamed conditions.

    reference_cue, comparison_cue : str or None
        Cue names, e.g. "ITD", "ILD", "COMBINED".

    reference_angle : float or None
        Reference angle. Matching is done on absolute value, so mirrored
        left/right versions are treated together.

    reference_center_frequency, comparison_center_frequency : float or None
        Frequencies in Hz.

    Returns
    -------
    matches : list of Path
        Raw CSV files matching the requested parameters.
    """

    subject_dir = Path(save_root) / subject_id

    if not subject_dir.exists():
        raise FileNotFoundError(f"No data directory found: {subject_dir}")

    csv_files = sorted(subject_dir.glob("*.csv"))
    matches = []

    for csv_path in csv_files:

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not read {csv_path}: {e}")
            continue

        if df.empty:
            continue

        keep = True

        if condition_id is not None:
            keep &= "condition_id" in df.columns
            keep &= (df["condition_id"].astype(str) == str(condition_id)).any()

        if reference_cue is not None:
            keep &= "reference_cue" in df.columns
            keep &= (df["reference_cue"].astype(str) == str(reference_cue)).any()

        if comparison_cue is not None:
            keep &= "comparison_cue" in df.columns
            keep &= (df["comparison_cue"].astype(str) == str(comparison_cue)).any()

        if reference_angle is not None:
            keep &= "reference_angle" in df.columns

            if keep:
                keep &= (
                    df["reference_angle"].astype(float).abs()
                    == abs(float(reference_angle))
                ).any()

        if reference_center_frequency is not None:
            keep &= "reference_center_frequency" in df.columns

            if keep:
                keep &= (
                    df["reference_center_frequency"].astype(float)
                    == float(reference_center_frequency)
                ).any()

        if comparison_center_frequency is not None:
            keep &= "comparison_center_frequency" in df.columns

            if keep:
                keep &= (
                    df["comparison_center_frequency"].astype(float)
                    == float(comparison_center_frequency)
                ).any()

        if keep:
            matches.append(csv_path)

    return matches


def fit_runs_by_params(
    subject_id,
    save_root="data/raw",
    derivatives_root="data/psychometrics",
    condition_id=None,
    reference_cue=None,
    comparison_cue=None,
    reference_angle=None,
    reference_center_frequency=None,
    comparison_center_frequency=None,
    overwrite=False,
):
    """
    Find matching raw files and fit all matching data together.

    This function does not run a new experiment. It only searches existing
    raw CSV files and fits the psychometric functions for the matching trials.

    Matching is based on metadata inside the CSV files.

    Returns
    -------
    summary : pandas.DataFrame
        Psychometric fit summary table for the matching data.
    """

    files = find_run_files(
        subject_id=subject_id,
        save_root=save_root,
        condition_id=condition_id,
        reference_cue=reference_cue,
        comparison_cue=comparison_cue,
        reference_angle=reference_angle,
        reference_center_frequency=reference_center_frequency,
        comparison_center_frequency=comparison_center_frequency,
    )

    if not files:
        print("No matching run files found.")
        return pd.DataFrame()

    summary = fit_run_files(
        csv_paths=files,
        derivatives_root=derivatives_root,
        sigmoid="norm",
        experiment_type="yes/no",
        overwrite=overwrite,
        analysis_label="combined_runs",
    )

    return summary


def fit_single_run_file(
    csv_path,
    derivatives_root="data/psychometrics",
    overwrite=False,
):
    """
    Fit psychometric functions from one specific raw CSV file.

    Useful for quick checks immediately after a run.
    """

    return fit_run_file(
        csv_path=csv_path,
        derivatives_root=derivatives_root,
        sigmoid="norm",
        experiment_type="yes/no",
        overwrite=overwrite,
    )