# experiment/two_afc.py

from dataclasses import dataclass
from typing import Optional, Union
import random
import time
from copy import deepcopy

from stimuli.sound_handler import generate_stim, apply_cue


@dataclass
class StimulusSpec:
    label: str
    cue: str
    center_frequency: float
    angle: Optional[float] = None
    value: Optional[Union[float, dict]] = None


@dataclass
class TrialSpec:
    standard: StimulusSpec
    comparison: StimulusSpec
    duration: float = 0.3
    samplerate: int = 44100
    level: float = 80
    isi: float = 0.2
    head_radius: float = 8.75


def make_stimulus(stim_spec, trial_spec, base_sound=None):
    """
    Generate and spatialise one stimulus.

    If base_sound is given, it is copied and used as the starting sound.
    This allows standard and comparison to share the same noise token.
    If base_sound is None, a new sound is generated.
    """

    if base_sound is None:
        sound, stim_info = generate_stim(
            center_frequency=stim_spec.center_frequency,
            duration=trial_spec.duration,
            samplerate=trial_spec.samplerate,
            level=trial_spec.level,
        )
    else:
        sound = deepcopy(base_sound)
        stim_info = {
            "center_frequency": stim_spec.center_frequency,
            "duration": trial_spec.duration,
            "samplerate": trial_spec.samplerate,
            "level": trial_spec.level
        }

    sound, cue_info = apply_cue(
        sound=sound,
        cue=stim_spec.cue,
        angle=stim_spec.angle,
        value=stim_spec.value,
        center_frequency=stim_spec.center_frequency,
        head_radius=trial_spec.head_radius,
    )

    info = {
        "label": stim_spec.label,
        "cue": stim_spec.cue,
        "angle": cue_info["angle"],
        "value": cue_info["value"],
        "center_frequency": stim_spec.center_frequency,
        **stim_info,
        **cue_info,
    }

    return sound, info


def prepare_2afc_trial(trial_spec, rng=None):
    """
    Prepare one 2AFC trial.

    """

    if rng is None:
        rng = random.Random()

    base_sound, _ = generate_stim(
        center_frequency=trial_spec.standard.center_frequency,
        duration=trial_spec.duration,
        samplerate=trial_spec.samplerate,
        level=trial_spec.level,
    )

    standard_sound, standard_info = make_stimulus(
        stim_spec=trial_spec.standard,
        trial_spec=trial_spec,
        base_sound=base_sound,
    )

    comparison_sound, comparison_info = make_stimulus(
        stim_spec=trial_spec.comparison,
        trial_spec=trial_spec,
        base_sound=base_sound,
    )

    order = ["standard", "comparison"]
    sounds = {
        "standard": standard_sound,
        "comparison": comparison_sound,
    }

    rng.shuffle(order)

    ordered_sounds = [sounds[order[0]], sounds[order[1]]]

    solution = get_solution(
        order=order,
        standard=standard_info,
        comparison=comparison_info,
    )

    trial_info = {
        "first": order[0],
        "second": order[1],
        "solution": solution,
        "standard": standard_info,
        "comparison": comparison_info,
        "isi": trial_spec.isi,
        "duration": trial_spec.duration,
        "samplerate": trial_spec.samplerate,
        "level": trial_spec.level,
        "head_radius": trial_spec.head_radius
    }

    return ordered_sounds, trial_info


def get_solution(order, standard, comparison):
    """
    Determine whether the second sound is further left or right
    than the first sound.

    Uses resolved angles from standard_info and comparison_info.
    """

    specs = {
        "standard": standard,
        "comparison": comparison,
    }

    first = specs[order[0]]
    second = specs[order[1]]

    first_angle = first["angle"]
    second_angle = second["angle"]

    if second_angle > first_angle:
        return "right"
    elif second_angle < first_angle:
        return "left"
    else:
        return "same"


def play_2afc_trial(ordered_sounds, isi=0.2):
    """
    Play two sounds with a silent interval in between.
    """

    first_sound, second_sound = ordered_sounds

    first_sound.play()
    time.sleep(isi)

    second_sound.play()