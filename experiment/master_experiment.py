from experiment.experiment_planner import plan_k_slope_run, plan_jnd_run
from experiment.run_builders import make_jnd_run, make_k_slope_run, make_practice_run
from experiment.experiment_pipeline import run_experiment_from_conditions, fit_runs_by_params
from analysis.estimate_k_slopes import estimate_k_slopes
from analysis.plot_k_slopes import plot_k_slopes
from analysis.plot_jnds import plot_jnds

# ============================================================
# General settings
# ============================================================

SUBJECT_ID = "sub-07"

FREQUENCY = 1400

# ============================================================
# 1. Practice run
# ============================================================

# practice_conditions = make_practice_run()
#
# run_experiment_from_conditions(
#     subject_id=SUBJECT_ID,
#     conditions=practice_conditions,
#     analysis_role="practice",
#     include_in_analysis=False,
# )


# ============================================================
# 2. JND run (repeat x3 for different JND cues)
# ============================================================

JND_CUE = "COMBINED"  # Use 'ITD', 'ILD', or 'COMBINED'

JND_COMPARISON_ANGLES = plan_jnd_run(
    cue=JND_CUE,
    frequency=FREQUENCY,
)

jnd_conditions = make_jnd_run(
    frequency=FREQUENCY,
    cue=JND_CUE,
    comparison_angles=JND_COMPARISON_ANGLES,
)

run_experiment_from_conditions(
    subject_id=SUBJECT_ID,
    conditions=jnd_conditions,
)

# ============================================================
# k slope run (repeat x3 for different k slope cues)
# ============================================================

# K_SLOPE_CUE = "COMBINED"  # Use 'ITD', 'ILD', or 'COMBINED'
#
# K_REFERENCE_POINTS = plan_k_slope_run(
#     reference_cue=K_SLOPE_CUE,
#     frequency=FREQUENCY,
# )

# Option to manually set up reference points

# K_REFERENCE_POINTS = [
#     {"reference_cue": K_SLOPE_CUE, "reference_angle": 4, "pse_estimate_value": 1},
#     {"reference_cue": K_SLOPE_CUE, "reference_angle": 8, "pse_estimate_value": 2},
#     {"reference_cue": K_SLOPE_CUE, "reference_angle": 12, "pse_estimate_value": 3},
# ]

# k_slope_conditions = make_k_slope_run(
#     frequency=FREQUENCY,
#     reference_points=K_REFERENCE_POINTS,
#     n_repetitions=4
# )
#
# run_experiment_from_conditions(
#     subject_id=SUBJECT_ID,
#     conditions=k_slope_conditions,
# )

# ============================================================
# Analysis
# ============================================================

summary = fit_runs_by_params(subject_id=SUBJECT_ID)
points, slopes = estimate_k_slopes(subject_id=SUBJECT_ID)

# ============================================================
# Plotting
# ============================================================
#
fig_k, axes_k = plot_k_slopes(subject_id=SUBJECT_ID)
fig_jnd, axes_jnd = plot_jnds(subject_id=SUBJECT_ID)

