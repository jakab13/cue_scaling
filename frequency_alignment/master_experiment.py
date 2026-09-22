from frequency_alignment.run_experiment import (
    run_low_ild,
    run_low_itd,
    run_combined,
    run_high_ild,
    run_all,
    fit_psychometrics,
    plot_results,
)


# ============================================================
# PARTICIPANT
# ============================================================

SUBJECT_ID = "jakab_test9"


# ============================================================
# EXPERIMENT
# ============================================================

run_low_ild(SUBJECT_ID)

run_low_itd(SUBJECT_ID)

run_combined(SUBJECT_ID)

run_high_ild(SUBJECT_ID)

# Or:
# run_all(SUBJECT_ID)


# ============================================================
# ANALYSIS
# ============================================================

fit_psychometrics(SUBJECT_ID)

plot_results(SUBJECT_ID)