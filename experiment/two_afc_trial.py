# experiment/two_afc.py

from dataclasses import dataclass
from typing import Optional, Union
import random
import time
from copy import deepcopy

import slab

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
    reference: StimulusSpec
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
    This allows reference and comparison to share the same noise token.
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

    ref_freq = trial_spec.reference.center_frequency
    comp_freq = trial_spec.comparison.center_frequency

    # --------------------------------------------------------
    # Generate the appropriate sound at each frequency
    # --------------------------------------------------------

    if ref_freq == comp_freq:

        # Same-frequency experiment:
        # share one noise token as before.
        base_sound, _ = generate_stim(
            center_frequency=ref_freq,
            duration=trial_spec.duration,
            samplerate=trial_spec.samplerate,
            level=trial_spec.level,
        )

        reference_base = base_sound
        comparison_base = base_sound

    else:

        # Across-frequency experiment:
        # each sound must be generated at its own frequency.
        reference_base, _ = generate_stim(
            center_frequency=ref_freq,
            duration=trial_spec.duration,
            samplerate=trial_spec.samplerate,
            level=trial_spec.level,
        )

        comparison_base, _ = generate_stim(
            center_frequency=comp_freq,
            duration=trial_spec.duration,
            samplerate=trial_spec.samplerate,
            level=trial_spec.level,
        )

    # --------------------------------------------------------
    # Apply spatial cues
    # --------------------------------------------------------

    reference_sound, reference_info = make_stimulus(
        stim_spec=trial_spec.reference,
        trial_spec=trial_spec,
        base_sound=reference_base,
    )

    comparison_sound, comparison_info = make_stimulus(
        stim_spec=trial_spec.comparison,
        trial_spec=trial_spec,
        base_sound=comparison_base,
    )

    # --------------------------------------------------------
    # Randomise presentation order
    # --------------------------------------------------------

    order = [
        "reference",
        "comparison",
    ]

    sounds = {
        "reference": reference_sound,
        "comparison": comparison_sound,
    }

    rng.shuffle(order)

    ordered_sounds = [
        sounds[order[0]],
        sounds[order[1]],
    ]

    solution = get_solution(
        order=order,
        reference=reference_info,
        comparison=comparison_info,
    )

    trial_info = {
        "first": order[0],
        "second": order[1],
        "solution": solution,
        "reference": reference_info,
        "comparison": comparison_info,
        "isi": trial_spec.isi,
        "duration": trial_spec.duration,
        "samplerate": trial_spec.samplerate,
        "level": trial_spec.level,
        "head_radius": trial_spec.head_radius,
    }

    return ordered_sounds, trial_info


def get_solution(order, reference, comparison):
    """
    Determine whether the second sound is further left or right
    than the first sound.

    Uses resolved angles from reference_info and comparison_info.
    """

    specs = {
        "reference": reference,
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

    silence = slab.Binaural.silence(duration=isi, samplerate=first_sound.samplerate)

    end_silence = slab.Binaural.silence(duration=isi, samplerate=first_sound.samplerate)

    full_trial = slab.Binaural.sequence(first_sound, silence, second_sound, end_silence)

    full_trial.play()
