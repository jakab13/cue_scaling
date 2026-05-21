from pathlib import Path
from datetime import datetime
import csv
import time

from experiment.two_afc_trial import prepare_2afc_trial, play_2afc_trial

def collect_response(
    trial_number,
    n_trials,
    left_key="l",
    right_key="r",
):
    """
    Collect a left/right response using input().

    Designed for a keyboard/button box that sends the keypress
    followed by Enter automatically.
    """

    prompt = (
        f"Trial {trial_number}/{n_trials} | "
        f"Response [{left_key}=left, {right_key}=right]: "
    )

    while True:
        key = input(prompt).strip().lower()

        if key in [left_key, "left"]:
            return "left"

        if key in [right_key, "right"]:
            return "right"

        print("Invalid response.")


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

    standard = trial_info["standard"]
    comparison = trial_info["comparison"]

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

        # standard stimulus
        "standard_angle": standard.get("angle"),
        "standard_cue": standard.get("cue"),
        "standard_center_frequency": standard.get("center_frequency"),
        "standard_ITD": standard.get("ITD"),
        "standard_ILD": standard.get("ILD"),

        # comparison stimulus
        "comparison_angle": comparison.get("angle"),
        "comparison_cue": comparison.get("cue"),
        "comparison_center_frequency": comparison.get("center_frequency"),
        "comparison_ITD": comparison.get("ITD"),
        "comparison_ILD": comparison.get("ILD"),

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
        "head_radius": trial_info["head_radius"]
    }

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
    run_label="run",
):
    """
    Create one unique CSV path per run.

    Example:
    data/raw/vp_001/vp_001_run_2026-05-20_11-42-03.csv
    """

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = Path(save_root) / subject_id
    save_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{subject_id}_{run_label}_{timestamp}.csv"

    return save_dir / filename

def run_experiment(
    subject_id,
    run_trials,
    save_root="data/raw",
    run_label="run",
    left_key="1",
    right_key="2"
):
    """
    Run one experimental sequence and save one CSV file for this run.
    """

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_path = make_run_save_path(
        subject_id=subject_id,
        save_root=save_root,
        run_label=run_label,
    )

    print("\nExperiment setup")
    print("-" * 40)
    print(f"Subject ID:      {subject_id}")
    print(f"Number of trials:{len(run_trials)}")
    print(f"Save file:       {save_path}")
    print("-" * 40)

    print("\nStarting run.")
    print("Response keys:")
    print(f"  {left_key} = left")
    print(f"  {right_key} = right")
    print()

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

        append_row_to_csv(
            row=row,
            save_path=save_path,
        )

    print(f"\nRun finished. Data saved to: {save_path}")

    return save_path