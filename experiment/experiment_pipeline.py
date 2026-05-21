# experiment/experiment_pipeline.py

from pathlib import Path
import pandas as pd

from experiment.trial_sequence import build_run_sequence
from experiment.run_experiment import run_experiment
from analysis.fit_psychometrics import fit_run_file


def run_and_fit_experiment(
    subject_id,
    conditions,
    run_label="run",
    save_root="data/raw",
    derivatives_root="data/psychometrics",
    randomize=True,
    seed=None,
    left_key="1",
    right_key="2",
    fit_after_run=True,
    overwrite_fit=False,
):
    """
    Build a run sequence, run the experiment, save one CSV file,
    and optionally fit psychometric functions immediately afterwards.
    """

    run_trials = build_run_sequence(
        conditions=conditions,
        randomize=randomize,
        seed=seed,
    )

    print("\nPrepared run")
    print("-" * 40)
    print(f"Subject ID: {subject_id}")
    print(f"Run label:  {run_label}")
    print(f"Conditions: {[c.condition_id for c in conditions]}")
    print(f"Trials:     {len(run_trials)}")
    print("-" * 40)

    csv_path = run_experiment(
        subject_id=subject_id,
        run_trials=run_trials,
        save_root=save_root,
        run_label=run_label,
        left_key=left_key,
        right_key=right_key,
    )

    if csv_path is None:
        print("No data file created. Skipping analysis.")
        return None, None

    if fit_after_run:
        print("\nFitting psychometric functions...")
        summary = fit_run_file(
            csv_path=csv_path,
            derivatives_root=derivatives_root,
            sigmoid="norm",
            experiment_type="yes/no",
            overwrite=overwrite_fit,
        )

        print("\nFit summary")
        print(summary[[
            "condition_id",
            "standard_cue",
            "standard_angle_abs",
            "comparison_cue",
            "PSE",
            "JND_84",
            "eta",
            "fit_loaded_from_cache",
        ]])

        return csv_path, summary

    return csv_path, None