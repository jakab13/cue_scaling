# experiment/trial_sequence.py

from dataclasses import dataclass
import random

from experiment.two_afc_trial import StimulusSpec, TrialSpec
from stimuli.sound_handler import angle_to_cue_value, value_to_angle


@dataclass
class ConditionSpec:
    """
    Defines one method-of-constant-stimuli condition.

    Comparison stimuli can be defined in two ways:

    1. Explicit list:
        comparison_angles=[...]
        or
        comparison_values=[...]

    2. PSE-centered list:
        pse_estimate_angle=...
        comparison_offsets=[...]

    In the PSE-centered case, offsets are interpreted in angle units.
    The generated comparison stimuli can still be presented as direct
    cue values by setting comparison_definition="value".
    """

    condition_id: str

    reference_cue: str
    reference_center_frequency: float
    reference_angle: float | None = None
    reference_value: float | dict | None = None

    comparison_cue: str = None
    comparison_center_frequency: float = None

    # Old explicit style
    comparison_angles: list | None = None
    comparison_values: list | None = None  # absolute cue values
    comparison_value_offsets: list | None = None  # relative to pse_estimate_value

    # New PSE-centered style
    pse_estimate_angle: float | None = None
    pse_estimate_value: float | dict | None = None
    comparison_offsets: list | None = None

    # Should generated comparison stimuli be angle-defined or value-defined?
    comparison_definition: str = "angle"  # "angle" or "value"

    n_repetitions: int = 4

    duration: float = 0.3
    samplerate: int = 44100
    level: float = 80
    isi: float = 0.2
    head_radius: float = 8.75

    mirror_trials: bool = True


@dataclass
class PlannedTrial:
    """
    A TrialSpec plus metadata useful for saving or debugging.
    """

    trial_index: int
    condition_id: str
    repetition: int
    comparison_index: int
    mirrored: bool
    trial_type: str
    trial_spec: TrialSpec
    pse_estimate_angle: float | None = None
    comparison_offset: float | None = None
    comparison_definition: str = "angle"



def make_trial_spec(
    reference_cue,
    comparison_cue,
    reference_center_frequency,
    comparison_center_frequency,
    reference_angle=None,
    reference_value=None,
    comparison_angle=None,
    comparison_value=None,
    duration=0.3,
    samplerate=44100,
    level=80,
    isi=0.2,
    head_radius=8.75,
):
    """
    Create one TrialSpec for a 2AFC presentation.
    """

    reference = StimulusSpec(
        label="reference",
        cue=reference_cue,
        center_frequency=reference_center_frequency,
        angle=reference_angle,
        value=reference_value,
    )

    comparison = StimulusSpec(
        label="comparison",
        cue=comparison_cue,
        center_frequency=comparison_center_frequency,
        angle=comparison_angle,
        value=comparison_value,
    )

    trial_spec = TrialSpec(
        reference=reference,
        comparison=comparison,
        duration=duration,
        samplerate=samplerate,
        level=level,
        isi=isi,
        head_radius=head_radius,
    )

    return trial_spec


def get_pse_estimate_angle(condition):
    """
    Return the PSE estimate in angle units.

    Preferred:
        condition.pse_estimate_angle

    If only pse_estimate_value is provided, infer the closest matching angle
    using the comparison cue mapping.
    """

    if condition.pse_estimate_angle is not None:
        return condition.pse_estimate_angle

    if condition.pse_estimate_value is not None:
        pse_angle, _ = value_to_angle(
            cue=condition.comparison_cue,
            value=condition.pse_estimate_value,
            center_frequency=condition.comparison_center_frequency,
            head_radius=condition.head_radius,
        )
        return pse_angle

    return None


def get_comparison_plan(condition):
    """
    Create the list of comparison points for one condition.

    Returns
    -------
    comparison_plan : list of dict
        Each dictionary contains:

        comparison_angle:
            Angle used to define the comparison stimulus.
            None if the comparison is value-defined.

        comparison_value:
            Cue value used to define the comparison stimulus.
            None if the comparison is angle-defined.

        comparison_offset:
            Offset relative to the PSE estimate, if applicable.

        comparison_definition:
            "angle" or "value".
    """

    # ------------------------------------------------------------
    # Case 1: absolute comparison angles
    # ------------------------------------------------------------
    if condition.comparison_angles is not None:

        comparison_plan = []

        for angle in condition.comparison_angles:
            comparison_plan.append(
                {
                    "comparison_angle": angle,
                    "comparison_value": None,
                    "comparison_offset": (
                        angle - condition.pse_estimate_angle
                        if condition.pse_estimate_angle is not None
                        else None
                    ),
                    "comparison_definition": "angle",
                }
            )

        return comparison_plan

    # ------------------------------------------------------------
    # Case 2: angle offsets around an angle-defined PSE
    # ------------------------------------------------------------
    if condition.comparison_offsets is not None:

        if condition.pse_estimate_angle is None:
            raise ValueError(
                "comparison_offsets require pse_estimate_angle."
            )

        comparison_plan = []

        for offset in condition.comparison_offsets:
            comparison_angle = condition.pse_estimate_angle + offset

            comparison_plan.append(
                {
                    "comparison_angle": comparison_angle,
                    "comparison_value": None,
                    "comparison_offset": offset,
                    "comparison_definition": "angle",
                }
            )

        return comparison_plan

    # ------------------------------------------------------------
    # Case 3: absolute comparison values
    # ------------------------------------------------------------
    if condition.comparison_values is not None:

        comparison_plan = []

        for value in condition.comparison_values:
            comparison_plan.append(
                {
                    "comparison_angle": None,
                    "comparison_value": value,
                    "comparison_offset": (
                        value - condition.pse_estimate_value
                        if condition.pse_estimate_value is not None
                        else None
                    ),
                    "comparison_definition": "value",
                }
            )

        return comparison_plan

    # ------------------------------------------------------------
    # Case 4: value offsets around a value-defined PSE
    # ------------------------------------------------------------
    if condition.comparison_value_offsets is not None:

        if condition.pse_estimate_value is None:
            raise ValueError(
                "comparison_value_offsets require pse_estimate_value."
            )

        comparison_plan = []

        for offset in condition.comparison_value_offsets:
            comparison_value = condition.pse_estimate_value + offset

            comparison_plan.append(
                {
                    "comparison_angle": None,
                    "comparison_value": comparison_value,
                    "comparison_offset": offset,
                    "comparison_definition": "value",
                }
            )

        return comparison_plan

    # ------------------------------------------------------------
    # No valid comparison definition
    # ------------------------------------------------------------
    raise ValueError(
        "No comparison points defined. Use one of: "
        "comparison_angles, comparison_offsets, "
        "comparison_values, or comparison_value_offsets."
    )


def mirror_value(value):
    """
    Mirror a direct cue value around the midline.
    """

    if value is None:
        return None

    if isinstance(value, dict):
        return {
            key: -val
            for key, val in value.items()
        }

    return -value


def build_condition_trials(condition, start_index=0):
    """
    Build all trials for one condition using the method of constant stimuli.

    For each comparison angle, the trial is repeated n_repetitions times.
    If mirror_trials=True, approximately half of the repetitions are mirrored
    around the midline by multiplying both reference and comparison angles by -1.

    Example:
    reference = 8, comparison = 20
    mirrored version: reference = -8, comparison = -20
    """

    planned_trials = []
    trial_index = start_index

    trial_type = f"{condition.reference_cue}-->{condition.comparison_cue}"
    comparison_plan = get_comparison_plan(condition)

    for comparison_index, comparison_item in enumerate(comparison_plan):

        n_mirrored = condition.n_repetitions // 2

        mirror_flags = (
                [True] * n_mirrored
                + [False] * (condition.n_repetitions - n_mirrored)
        )

        for repetition, mirrored in enumerate(mirror_flags):

            reference_angle = condition.reference_angle
            reference_value = condition.reference_value

            comparison_angle = comparison_item["comparison_angle"]
            comparison_value = comparison_item["comparison_value"]

            if condition.mirror_trials and mirrored:
                reference_angle = (
                    -reference_angle
                    if reference_angle is not None
                    else None
                )

                if reference_value is not None:
                    reference_value = -reference_value

                comparison_angle = (
                    -comparison_angle
                    if comparison_angle is not None
                    else None
                )

                if comparison_value is not None:
                    comparison_value = -comparison_value

            trial_spec = make_trial_spec(
                reference_cue=condition.reference_cue,
                comparison_cue=condition.comparison_cue,
                reference_center_frequency=condition.reference_center_frequency,
                comparison_center_frequency=condition.comparison_center_frequency,
                reference_angle=reference_angle,
                reference_value=reference_value,
                comparison_angle=comparison_angle,
                comparison_value=comparison_value,
                duration=condition.duration,
                samplerate=condition.samplerate,
                level=condition.level,
                isi=condition.isi,
                head_radius=condition.head_radius,
            )

            planned_trial = PlannedTrial(
                trial_index=trial_index,
                condition_id=condition.condition_id,
                repetition=repetition,
                comparison_index=comparison_index,
                mirrored=condition.mirror_trials and mirrored,
                trial_type=trial_type,
                trial_spec=trial_spec,
            )

            planned_trials.append(planned_trial)
            trial_index += 1

    return planned_trials


def build_run_sequence(
    conditions,
    randomize=True,
    seed=None,
):
    """
    Build a full randomized run sequence from one or more ConditionSpec objects.
    """

    rng = random.Random(seed)

    all_trials = []
    trial_index = 0

    for condition in conditions:
        condition_trials = build_condition_trials(
            condition=condition,
            start_index=trial_index,
        )

        all_trials.extend(condition_trials)
        trial_index += len(condition_trials)

    if randomize:
        rng.shuffle(all_trials)

    # Re-index after shuffling so trial_index reflects actual presentation order.
    # Also recompute repetition so it reflects the order in which each
    # condition/comparison/mirror cell appears during the actual run.
    reindexed_trials = []

    repetition_counter = {}

    for new_index, planned_trial in enumerate(all_trials):
        repetition_key = (
            planned_trial.condition_id,
            planned_trial.comparison_index
        )

        current_repetition = repetition_counter.get(repetition_key, 0)

        planned_trial.trial_index = new_index
        planned_trial.repetition = current_repetition

        repetition_counter[repetition_key] = current_repetition + 1

        reindexed_trials.append(planned_trial)

    return reindexed_trials

