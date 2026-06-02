from experiment.experiment_planner import plan_k_slope_run, plan_jnd_run
from experiment.run_builders import make_jnd_run, make_k_slope_run
from experiment.experiment_pipeline import run_experiment_from_conditions, fit_runs_by_params
from analysis.estimate_k_slopes import estimate_k_slopes
from analysis.plot_k_slopes import plot_k_slopes
from analysis.plot_jnds import plot_jnds

# ============================================================
# General settings
# ============================================================

SUBJECT_ID = "jakab"

FREQUENCY = 1000

K_SLOPE_CUE = "COMBINED"  # Use 'ITD', 'ILD', or 'COMBINED'

JND_CUE = "COMBINED"

JND_COMPARISON_ANGLES = plan_jnd_run(
    cue=JND_CUE,
    frequency=FREQUENCY,
)

jnd_conditions = make_jnd_run(
    frequency=FREQUENCY,
    cue=JND_CUE,
    comparison_angles=JND_COMPARISON_ANGLES,
)

practice_conditions = make_jnd_run(
    frequency=1000,
    cue="COMBINED",
    comparison_angles=[-21, -15, -9, -3, 3, 9, 15, 21],
)

run_experiment_from_conditions(
    subject_id=SUBJECT_ID,
    conditions=practice_conditions,
    analysis_role="practice",
    include_in_analysis=False,
)

# ============================================================
# k-slope run settings
# Used only if RUN_MODE == "K_SLOPE"
# ============================================================

# Each point is one PSE measurement.
# pse_estimate_value and comparison_value_offsets are in comparison-cue units.
# For ILD, that means dB.

K_REFERENCE_POINTS = plan_k_slope_run(
    subject_id=SUBJECT_ID,
    reference_cue=K_SLOPE_CUE,
    frequency=FREQUENCY,
)

# K_REFERENCE_POINTS = [
#     {"reference_cue": K_SLOPE_CUE, "reference_angle": 4, "pse_estimate_value": 1},
#     {"reference_cue": K_SLOPE_CUE, "reference_angle": 8, "pse_estimate_value": 2},
#     {"reference_cue": K_SLOPE_CUE, "reference_angle": 12, "pse_estimate_value": 3},
# ]

k_slope_conditions = make_k_slope_run(
    frequency=FREQUENCY,
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
summary = fit_runs_by_params(subject_id=SUBJECT_ID)

points, slopes = estimate_k_slopes(save=True)
fig, axes = plot_k_slopes(subject_id=SUBJECT_ID)
# fig, axes = plot_jnds(subject_id=SUBJECT_ID)

