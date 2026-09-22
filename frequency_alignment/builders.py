from experiment.trial_sequence import ConditionSpec


DEFAULT_REPETITIONS = 8


def make_alignment_condition(
    condition_id,
    reference_frequency,
    comparison_frequency,
    cue,
    reference_angle,
    comparison_angles,
    n_repetitions=DEFAULT_REPETITIONS,
):
    """
    Create one across-frequency alignment condition.
    """

    cue = cue.upper()

    return ConditionSpec(
        condition_id=condition_id,

        reference_cue=cue,
        reference_center_frequency=reference_frequency,
        reference_angle=reference_angle,

        comparison_cue=cue,
        comparison_center_frequency=comparison_frequency,
        comparison_angles=comparison_angles,
        comparison_definition="angle",

        n_repetitions=n_repetitions,
        mirror_trials=True,
    )


def make_low_ild_block(
    n_repetitions=DEFAULT_REPETITIONS,
):
    return [
        make_alignment_condition(
            condition_id="low_ild_1400to1000",
            reference_frequency=1400,
            comparison_frequency=1000,
            cue="ILD",
            reference_angle=20,
            comparison_angles=[
                -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40,
            ],
            n_repetitions=n_repetitions,
        )
    ]


def make_low_itd_block(
    n_repetitions=DEFAULT_REPETITIONS,
):
    return [
        make_alignment_condition(
            condition_id="low_itd_1400to1000",
            reference_frequency=1400,
            comparison_frequency=1000,
            cue="ITD",
            reference_angle=20,
            comparison_angles=[
                -5, 0, 5, 10, 15, 20, 25,
            ],
            n_repetitions=n_repetitions,
        )
    ]


def make_combined_block(
    n_repetitions=DEFAULT_REPETITIONS,
):
    return [
        make_alignment_condition(
            condition_id="combined_1400to1000",
            reference_frequency=1400,
            comparison_frequency=1000,
            cue="COMBINED",
            reference_angle=20,
            comparison_angles=[
                -10, -5, 0, 5, 10, 15, 20
            ],
            n_repetitions=n_repetitions,
        )
    ]


def make_high_ild_block(
    n_repetitions=DEFAULT_REPETITIONS,
):
    return [
        make_alignment_condition(
            condition_id="high_ild_1400to4000",

            reference_frequency=1400,
            comparison_frequency=4000,

            cue="ILD",

            reference_angle=20,

            comparison_angles=[
                -10, -5, 0, 5, 10, 15, 20
            ],

            n_repetitions=n_repetitions,
        )
    ]