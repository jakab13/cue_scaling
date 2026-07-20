# experiment/run_builders.py

from experiment.trial_sequence import ConditionSpec


DEFAULT_LEVEL = 80
DEFAULT_DURATION = 0.3
DEFAULT_ISI = 0.2
DEFAULT_HEAD_RADIUS = 8.75

DEFAULT_JND_REPETITIONS = 16
DEFAULT_K_SLOPE_REPETITIONS = 8

DEFAULT_K_COMPARISON_CUE = "ILD"

# For ILD comparison measurements, these are in dB.
DEFAULT_K_COMPARISON_VALUE_OFFSETS = [
    -7.0, -5.0, -3.0, -1.0,
    1.0, 3.0, 5.0, 7.0
]

DEFAULT_MIN_COMPARISON_VALUE = -20
DEFAULT_MAX_COMPARISON_VALUE = 20

def make_jnd_run(
    frequency,
    cue,
    comparison_angles,
    n_repetitions=DEFAULT_JND_REPETITIONS,
    level=DEFAULT_LEVEL,
    duration=DEFAULT_DURATION,
    isi=DEFAULT_ISI,
    head_radius=DEFAULT_HEAD_RADIUS,
):
    """
    Create a centered JND run for one cue at one frequency.
    """

    cue = cue.upper()

    return [
        ConditionSpec(
            condition_id=f"jnd_{cue.lower()}_{int(frequency)}",

            reference_cue=cue,
            reference_center_frequency=frequency,
            reference_angle=0,

            comparison_cue=cue,
            comparison_center_frequency=frequency,
            comparison_angles=comparison_angles,
            comparison_definition="angle",

            n_repetitions=n_repetitions,
            mirror_trials=False,

            level=level,
            duration=duration,
            isi=isi,
            head_radius=head_radius,
        )
    ]


def make_k_slope_run(
    frequency,
    reference_points,
    comparison_cue=DEFAULT_K_COMPARISON_CUE,
    n_repetitions=DEFAULT_K_SLOPE_REPETITIONS,
    comparison_value_offsets=None,
    level=DEFAULT_LEVEL,
    duration=DEFAULT_DURATION,
    isi=DEFAULT_ISI,
    head_radius=DEFAULT_HEAD_RADIUS,
    enforce_value_limits=True,
    min_value=DEFAULT_MIN_COMPARISON_VALUE,
    max_value=DEFAULT_MAX_COMPARISON_VALUE,
):
    """
    Create a k-slope/PSE-measurement run for one frequency.

    The master file only needs to define:
        - frequency
        - reference cue
        - reference angle
        - PSE estimate value

    comparison_value_offsets are shared across all k-slope conditions
    unless explicitly overridden.
    """

    if comparison_value_offsets is None:
        comparison_value_offsets = DEFAULT_K_COMPARISON_VALUE_OFFSETS

    comparison_cue = comparison_cue.upper()
    conditions = []

    for point in reference_points:

        reference_cue = point["reference_cue"].upper()
        reference_angle = point["reference_angle"]
        pse_estimate_value = point["pse_estimate_value"]

        # Optional per-point override, but not needed in normal use
        point_offsets = point.get(
            "comparison_value_offsets",
            comparison_value_offsets,
        )

        comparison_values = [
            pse_estimate_value + offset
            for offset in point_offsets
        ]

        if enforce_value_limits:
            below = [v for v in comparison_values if v < min_value]
            above = [v for v in comparison_values if v > max_value]

            if below or above:
                raise ValueError(
                    f"Comparison values out of allowed range for "
                    f"{reference_cue} ref {reference_angle}° at {frequency} Hz.\n"
                    f"Allowed range: {min_value} to {max_value}\n"
                    f"Generated values: {comparison_values}\n"
                    f"PSE estimate: {pse_estimate_value}\n"
                    f"Offsets: {point_offsets}"
                )

        condition = ConditionSpec(
            condition_id=(
                f"kslope_{reference_cue.lower()}_to_"
                f"{comparison_cue.lower()}_"
                f"{int(frequency)}_ref{reference_angle}"
            ),

            reference_cue=reference_cue,
            reference_center_frequency=frequency,
            reference_angle=reference_angle,

            comparison_cue=comparison_cue,
            comparison_center_frequency=frequency,

            pse_estimate_value=pse_estimate_value,
            comparison_value_offsets=point_offsets,
            comparison_definition="value",

            n_repetitions=n_repetitions,
            mirror_trials=True,

            level=level,
            duration=duration,
            isi=isi,
            head_radius=head_radius,
        )

        conditions.append(condition)

    return conditions


def make_practice_run(
    frequency=1000,
    cue="COMBINED",
    comparison_angles=None,
    n_repetitions=8,
):
    """
    Short centred JND run for practice/test purposes.
    Not intended for main analysis.
    """

    if comparison_angles is None:
        comparison_angles = [-35, -25, -15, -5, 5, 15, 25, 35]

    return make_jnd_run(
        frequency=frequency,
        cue=cue,
        comparison_angles=comparison_angles,
        n_repetitions=n_repetitions,
    )

