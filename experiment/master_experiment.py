from experiment.trial_sequence import ConditionSpec
from experiment.experiment_pipeline import run_and_fit_experiment


# ============================================================
# Experimenter settings
# ============================================================

SUBJECT_ID = "jakab"
CENTER_FREQUENCY = 1400
STANDARD_CUE = "ITD"
STANDARD_ANGLE = 15
PSE_ESTIMATE_VALUE = 1  # in dB

# ============================================================
# Define run conditions
# ============================================================

conditions = [

    ConditionSpec(
        condition_id=f"{STANDARD_CUE}_to_ILD",

        standard_cue=STANDARD_CUE,
        standard_center_frequency=CENTER_FREQUENCY,
        standard_angle=STANDARD_ANGLE,

        comparison_cue="ILD",
        comparison_center_frequency=CENTER_FREQUENCY,

        pse_estimate_value=PSE_ESTIMATE_VALUE,
        comparison_value_offsets=[
            -3.5, -2.5, -1.5, -0.5,
             0.5,  1.5,  2.5,  3.5,
        ],

        comparison_definition="value",

        n_repetitions=8,
    ),

]


# ============================================================
# Run experiment and fit immediately
# ============================================================

if __name__ == "__main__":

    csv_path, summary = run_and_fit_experiment(
        subject_id=SUBJECT_ID,
        conditions=conditions,
        fit_after_run=True,
        overwrite_fit=False,
    )