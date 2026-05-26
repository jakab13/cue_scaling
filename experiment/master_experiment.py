# TODO more console feedback for experimenter (e.g. "Calculating psychometric fit")
# TODO plot psychometric fits in subplots (cue x standard_angle_abs)

# master_experiment.py

from experiment.run_builders import make_jnd_run, make_k_slope_run
from experiment.experiment_pipeline import run_experiment_from_conditions, fit_runs_by_params
from analysis.view_psychometrics import show_psychometric_figures


# ============================================================
# General settings
# ============================================================

SUBJECT_ID = "jakab"

FREQUENCY = 1400

RUN_MODE = "K_SLOPE"
K_SLOPE_CUE = "ITD"

# JND_CUE = "ILD"
#
# JND_COMPARISON_ANGLES = [
#     -6, -4, -2, -1,
#      1,  2,  4,  6,
# ]

# jnd_conditions = make_jnd_run(
#     frequency=FREQUENCY,
#     cue=JND_CUE,
#     comparison_angles=JND_COMPARISON_ANGLES,
# )

# ============================================================
# k-slope run settings
# Used only if RUN_MODE == "K_SLOPE"
# ============================================================

# Each point is one PSE measurement.
# pse_estimate_value and comparison_value_offsets are in comparison-cue units.
# For ILD, that means dB.
K_REFERENCE_POINTS = [
    {"reference_cue": K_SLOPE_CUE, "reference_angle": 5, "pse_estimate_value": 1.0},
    {"reference_cue": K_SLOPE_CUE, "reference_angle": 10, "pse_estimate_value": 1.5},
    {"reference_cue": K_SLOPE_CUE, "reference_angle": 15, "pse_estimate_value": 2.},
]

k_slope_conditions = make_k_slope_run(
    frequency=1400,
    reference_points=K_REFERENCE_POINTS
)
# ============================================================
# Console commands
# ============================================================

# Run a JND run:
# run_experiment_from_conditions(
#     subject_id=SUBJECT_ID,
#     conditions=jnd_conditions,
# )

# Run a k-slope run:
run_experiment_from_conditions(
    subject_id=SUBJECT_ID,
    conditions=k_slope_conditions,
)

# Fit existing data by parameters:
summary = fit_runs_by_params(
    subject_id=SUBJECT_ID,
    reference_cue=K_SLOPE_CUE,
    comparison_cue="ILD",
    reference_angle=15,
    reference_center_frequency=FREQUENCY,
    comparison_center_frequency=FREQUENCY,
)

# View existing psychometric figures:
show_psychometric_figures(
    subject_id=SUBJECT_ID,
    reference_cue=K_SLOPE_CUE,
    comparison_cue="ILD",
    reference_angle=5,
    reference_center_frequency=FREQUENCY,
    comparison_center_frequency=FREQUENCY,
)