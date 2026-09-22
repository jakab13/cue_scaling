from frequency_alignment.run_experiment import (
    run_alignment_block,
    fit_psychometrics,
    plot_results,
)


# ============================================================
# PARTICIPANT
# ============================================================

SUBJECT_ID = "jakab_test_10"


# ============================================================
# EXPERIMENT
# ============================================================

# Low-frequency ILD

run_alignment_block(
    subject_id=SUBJECT_ID,
    cue="ILD",
    reference_frequency=1400,
    comparison_frequency=1000,
    reference_angle=20,
    comparison_angles=[
        -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40,
    ],
    n_repetitions=8,
)


# Low-frequency ITD

run_alignment_block(
    subject_id=SUBJECT_ID,
    cue="ITD",
    reference_frequency=1400,
    comparison_frequency=1000,
    reference_angle=20,
    comparison_angles=[
        -5, 0, 5, 10, 15, 20, 25,
    ],
    n_repetitions=8,
)


# Combined cues

run_alignment_block(
    subject_id=SUBJECT_ID,
    cue="COMBINED",
    reference_frequency=1400,
    comparison_frequency=1000,
    reference_angle=20,
    comparison_angles=[
        -10, -5, 0, 5, 10, 15, 20,
    ],
    n_repetitions=8,
)


# High-frequency ILD

run_alignment_block(
    subject_id=SUBJECT_ID,
    cue="ILD",
    reference_frequency=1400,
    comparison_frequency=4000,
    reference_angle=20,
    comparison_angles=[
        -10, -5, 0, 5, 10, 15, 20,
    ],
    n_repetitions=8,
)


# ============================================================
# ANALYSIS
# ============================================================

fit_psychometrics(
    SUBJECT_ID
)

plot_results(
    SUBJECT_ID
)