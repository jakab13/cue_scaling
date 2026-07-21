from pathlib import Path
from datetime import datetime
import csv
import time
import numpy as np
from readchar import readkey

from experiment.two_afc_trial import prepare_2afc_trial, play_2afc_trial

def collect_response(
    trial_number,
    n_trials,
    left_key="1",
    right_key="2",
):
    """
    Collect a left/right response from a single keypress.

    No Enter press is required.
    """

    left_key = str(left_key).lower()
    right_key = str(right_key).lower()

    if len(left_key) != 1 or len(right_key) != 1:
        raise ValueError("left_key and right_key must each be one character.")

    prompt = (
        f"[Trial {trial_number}/{n_trials}] "
        f"Response [{left_key}=left, {right_key}=right]: "
    )

    print(prompt, end="", flush=True)

    while True:
        pressed_key = readkey().lower()

        # Silently ignore Enter. This also makes the function compatible
        # with button boxes that may send Enter after the response key.
        if pressed_key in ("\r", "\n"):
            continue

        if pressed_key == left_key:
            print("left")
            return "left"

        if pressed_key == right_key:
            print("right")
            return "right"

        print(f"{pressed_key!r} is invalid.")
        print(prompt, end="", flush=True)


def get_scalar_cue_value(stimulus_info):
    """
    Return the scalar cue value for value-based fits.

    ILD -> dB
    ITD -> seconds
    COMBINED -> None, because there is no single scalar value.
    """

    cue = str(stimulus_info.get("cue", "")).upper()

    if cue == "ILD":
        return stimulus_info.get("ILD")

    if cue == "ITD":
        return stimulus_info.get("ITD")

    return None


def add_folded_trial_columns(row):
    """
    Add analysis-ready folded columns to one saved trial row.
    """

    reference_angle = float(row["reference_angle"])

    fold_sign = 1 if reference_angle == 0 else np.sign(reference_angle)

    row["reference_angle_folded"] = abs(reference_angle)

    if row["comparison_angle"] is not None:
        row["comparison_angle_folded"] = float(row["comparison_angle"]) * fold_sign
    else:
        row["comparison_angle_folded"] = None

    if row["reference_value"] is not None:
        row["reference_value_folded"] = float(row["reference_value"]) * fold_sign
    else:
        row["reference_value_folded"] = None

    if row["comparison_value"] is not None:
        row["comparison_value_folded"] = float(row["comparison_value"]) * fold_sign
    else:
        row["comparison_value_folded"] = None

    return row


def flatten_trial_data(
    participant_id,
    run_id,
    planned_trial,
    trial_info,
    response,
    reaction_time,
):
    """
    Convert trial metadata and response into one flat row for CSV saving.
    """

    reference = trial_info["reference"]
    comparison = trial_info["comparison"]

    reference_value = get_scalar_cue_value(reference)
    comparison_value = get_scalar_cue_value(comparison)

    solution = trial_info["solution"]
    is_correct = response == solution

    row = {
        # participant/run
        "subject_id": participant_id,
        "run_id": run_id,
        "datetime": datetime.now().isoformat(),

        # planned trial metadata
        "trial_index": planned_trial.trial_index,
        "condition_id": planned_trial.condition_id,
        "trial_type": planned_trial.trial_type,
        "repetition": planned_trial.repetition,
        "comparison_index": planned_trial.comparison_index,
        "mirrored": planned_trial.mirrored,

        # reference stimulus
        "reference_angle": reference.get("angle"),
        "reference_cue": reference.get("cue"),
        "reference_center_frequency": reference.get("center_frequency"),
        "reference_ITD": reference.get("ITD"),
        "reference_ILD": reference.get("ILD"),
        "reference_value": reference_value,

        # comparison stimulus
        "comparison_angle": comparison.get("angle"),
        "comparison_cue": comparison.get("cue"),
        "comparison_center_frequency": comparison.get("center_frequency"),
        "comparison_ITD": comparison.get("ITD"),
        "comparison_ILD": comparison.get("ILD"),
        "comparison_value": comparison_value,

        # initial pse estimate
        "pse_estimate_angle": planned_trial.pse_estimate_angle,
        "comparison_offset": planned_trial.comparison_offset,
        "comparison_definition": planned_trial.comparison_definition,

        # presentation order
        "first": trial_info["first"],
        "second": trial_info["second"],

        # response
        "solution": solution,
        "response": response,
        "is_correct": is_correct,
        "reaction_time": reaction_time,

        # sound parameters
        "isi": trial_info["isi"],
        "duration": trial_info["duration"],
        "samplerate": trial_info["samplerate"],
        "level": trial_info["level"],
        "head_radius": trial_info["head_radius"],
    }

    row = add_folded_trial_columns(row)

    return row


def append_row_to_csv(row, save_path):
    """
    Append one trial row to a CSV file.
    Creates the file and header if needed.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = save_path.exists()

    with open(save_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def make_run_save_path(
    subject_id,
    save_root="data/raw",
):
    """
    Create one unique CSV path per run.

    Example:
        data/raw/jakab/jakab_2026-05-21_16-31-02.csv
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = Path(save_root) / subject_id
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{subject_id}_{timestamp}.csv"

    return save_dir / filename


def run_experiment(
    subject_id,
    run_trials,
    save_root="data/raw",
    left_key="1",
    right_key="2",
    analysis_role="main",
    include_in_analysis=True,
):
    """
    Run one experimental sequence and save one CSV file for this run.
    """

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_path = make_run_save_path(
        subject_id=subject_id,
        save_root=save_root
    )

    input("Ready?")

    print("\nStarting run.")

    for trial_number, planned_trial in enumerate(run_trials, start=1):

        ordered_sounds, trial_info = prepare_2afc_trial(
            planned_trial.trial_spec
        )

        play_2afc_trial(
            ordered_sounds=ordered_sounds,
            isi=planned_trial.trial_spec.isi,
        )

        response_start = time.time()

        response = collect_response(
            trial_number=trial_number,
            n_trials=len(run_trials),
            left_key=left_key,
            right_key=right_key,
        )

        reaction_time = time.time() - response_start

        row = flatten_trial_data(
            participant_id=subject_id,
            run_id=run_id,
            planned_trial=planned_trial,
            trial_info=trial_info,
            response=response,
            reaction_time=reaction_time,
        )

        row["analysis_role"] = analysis_role
        row["include_in_analysis"] = include_in_analysis

        append_row_to_csv(
            row=row,
            save_path=save_path,
        )

    print(f"\nRun finished. Data saved to: {save_path}")

    return save_path