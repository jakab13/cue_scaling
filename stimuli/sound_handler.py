import slab
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_ILS_PATH = Path("stimuli/ils_kemar.pickle")

def get_interaural_level_spectrum(
    save_path=DEFAULT_ILS_PATH,
    hrtf=None,
    overwrite=False,
):
    """
    Load an interaural level spectrum if it exists.
    Otherwise generate it with slab, save it, and return it.

    Parameters
    ----------
    save_path : str or Path
        Where the ILS file should be saved/loaded from.
    hrtf : None or slab.HRTF
        HRTF used to compute the ILS. If None, slab uses the default KEMAR HRTF.
    overwrite : bool
        If True, regenerate the ILS even if the file already exists.

    Returns
    -------
    ils : dict
        Interaural level spectrum dictionary.
    """

    # Function-level cache
    if not overwrite and hasattr(get_interaural_level_spectrum, "_cache"):
        return get_interaural_level_spectrum._cache

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not overwrite:
        # print(f"Loading existing ILS from: {save_path}")
        with open(save_path, "rb") as f:
            ils = pickle.load(f)
    else:
        print("Generating interaural level spectrum...")
        ils = slab.Binaural.make_interaural_level_spectrum(hrtf=hrtf)

        print(f"Saving ILS to: {save_path}")
        with open(save_path, "wb") as f:
            pickle.dump(ils, f)

    get_interaural_level_spectrum._cache = ils

    return ils


def ensure_binaural(sound):
    """
    Make sure the sound is a slab.Binaural object.

    If the input is a mono slab.Sound, this creates a diotic binaural version.
    If it is already slab.Binaural, it is returned unchanged.
    """
    if isinstance(sound, slab.Binaural):
        return sound

    if isinstance(sound, slab.Sound):
        return slab.Binaural(sound)

    raise TypeError(
        "sound must be a slab.Sound or slab.Binaural object, "
        f"but got {type(sound)}"
    )


def ild_tuple_to_scalar(ild):
    """
    Convert slab's ILD output to a single signed ILD value.

    slab may return a tuple/list with channel-specific level changes:
    (left_level, right_level).

    We define scalar ILD as:
        right_level - left_level

    This means:
        positive ILD = right ear louder
        negative ILD = left ear louder
    """

    if isinstance(ild, (tuple, list)):
        left_level, right_level = ild
        return right_level - left_level

    return ild


DEFAULT_ILD_SLOPE_AZIS = np.arange(-15, 20, 5, dtype=float)


def ild_from_ils(
    azis_deg,
    freq_hz,
    ils_dict=None,
):
    """
    Get scalar ILD values from the interaural level spectrum.

    Scalar ILD is defined as:
        right_level - left_level

    Parameters
    ----------
    azis_deg : float or array-like
        Azimuth angle(s) in degrees.

    freq_hz : float
        Frequency in Hz.

    ils_dict : dict or None
        Interaural level spectrum. If None, the default saved ILS is loaded.

    Returns
    -------
    ild_vals : np.ndarray
        ILD values in dB.
    """

    if ils_dict is None:
        ils_dict = get_interaural_level_spectrum()

    scalar_input = np.isscalar(azis_deg)

    if scalar_input:
        azis_deg = [azis_deg]

    ild_vals = []

    for azimuth in azis_deg:
        ild_tuple = slab.Binaural.azimuth_to_ild(
            azimuth=float(azimuth),
            frequency=freq_hz,
            ils=ils_dict,
        )

        ild_scalar = ild_tuple_to_scalar(ild_tuple)
        ild_vals.append(ild_scalar)

    ild_vals = np.asarray(ild_vals, dtype=float)

    if scalar_input:
        return float(ild_vals[0])

    return ild_vals


def ild_slope_at_zero_fit(
    freq_hz,
    ils_dict=None,
    azis_deg=DEFAULT_ILD_SLOPE_AZIS,
    return_abs=True,
):
    """
    Local linear-fit estimate of the slope of ILD(azimuth) around 0 degrees.

    Returns slope in dB/degree.

    Parameters
    ----------
    freq_hz : float
        Frequency in Hz.

    ils_dict : dict or None
        Interaural level spectrum. If None, the default saved ILS is loaded.

    azis_deg : array-like
        Azimuths used for the local linear fit.

    return_abs : bool
        If True, return the absolute slope magnitude.
        This is usually what we want for experiment planning.
    """

    if ils_dict is None:
        ils_dict = get_interaural_level_spectrum()

    azis_deg = np.asarray(azis_deg, dtype=float)

    ild_vals = ild_from_ils(
        azis_deg=azis_deg,
        freq_hz=freq_hz,
        ils_dict=ils_dict,
    )

    slope, intercept = np.polyfit(azis_deg, ild_vals, deg=1)

    if return_abs:
        return float(abs(slope))

    return float(slope)


def get_ILD_slopes(
    freqs_of_interest=(400, 600, 800, 1000, 1200, 1400),
    azis_deg=DEFAULT_ILD_SLOPE_AZIS,
    ils_dict=None,
    return_abs=True,
):
    """
    Estimate ILD slopes for multiple frequencies.

    Returns
    -------
    df_slopes : pandas.DataFrame
        Columns:
            standard_center_frequency
            slope_db
    """

    if ils_dict is None:
        ils_dict = get_interaural_level_spectrum()

    df_slopes = pd.DataFrame(
        {
            "standard_center_frequency": freqs_of_interest,
            "slope_db": [
                ild_slope_at_zero_fit(
                    freq_hz=f,
                    ils_dict=ils_dict,
                    azis_deg=azis_deg,
                    return_abs=return_abs,
                )
                for f in freqs_of_interest
            ],
        }
    )

    return df_slopes


def value_to_angle(
    cue,
    value,
    center_frequency,
    head_radius=8.75,
    ils=None,
    angle_range=(-90, 90),
    step=0.1,
):
    """
    Find the best-matching azimuth angle for a given cue value.

    This is used when a stimulus is directly defined by ITD or ILD value,
    but we still want an equivalent angle for saving and response scoring.

    Parameters
    ----------
    cue : {"ITD", "ILD", "COMBINED"}
        Cue type.

    value : float or dict
        Direct cue value.
        For ITD: scalar ITD.
        For ILD: scalar ILD in dB.
        For COMBINED: {"ITD": ..., "ILD": ...}

    center_frequency : float
        Center frequency in Hz.

    head_radius : float
        Head radius in cm, used for ITD conversion.

    ils : dict or None
        Interaural level spectrum, used for ILD conversion.

    angle_range : tuple
        Candidate angle range in degrees.

    step : float
        Resolution of the angle search in degrees.

    Returns
    -------
    best_angle : float
        Candidate angle whose cue value best matches the requested value.

    best_error : float
        Absolute error between requested and candidate cue value.
    """

    cue = cue.upper()

    if cue in ["ILD", "COMBINED"] and ils is None:
        ils = get_interaural_level_spectrum()

    candidate_angles = np.arange(
        angle_range[0],
        angle_range[1] + step,
        step,
    )

    errors = []

    for angle in candidate_angles:

        if cue == "ITD":
            candidate_value = slab.Binaural.azimuth_to_itd(
                azimuth=angle,
                frequency=center_frequency,
                head_radius=head_radius,
            )

            error = abs(candidate_value - value)

        elif cue == "ILD":
            candidate_ild = slab.Binaural.azimuth_to_ild(
                azimuth=angle,
                frequency=center_frequency,
                ils=ils,
            )

            candidate_value = ild_tuple_to_scalar(candidate_ild)
            error = abs(candidate_value - value)

        elif cue == "COMBINED":
            if not isinstance(value, dict):
                raise TypeError(
                    "For cue='COMBINED', value must be a dict, "
                    "for example {'ITD': 0.00018, 'ILD': 4.5}."
                )

            candidate_itd = slab.Binaural.azimuth_to_itd(
                azimuth=angle,
                frequency=center_frequency,
                head_radius=head_radius,
            )

            candidate_ild = slab.Binaural.azimuth_to_ild(
                azimuth=angle,
                frequency=center_frequency,
                ils=ils,
            )
            candidate_ild = ild_tuple_to_scalar(candidate_ild)

            itd_error = abs(candidate_itd - value.get("ITD", 0))
            ild_error = abs(candidate_ild - value.get("ILD", 0))

            # Simple combined mismatch.
            # This is mostly for bookkeeping; for direct COMBINED values,
            # exact inversion may not be meaningful.
            error = itd_error + ild_error

        else:
            raise ValueError(
                f"Unknown cue '{cue}'. Use 'ITD', 'ILD', or 'COMBINED'."
            )

        errors.append(error)

    best_index = int(np.argmin(errors))
    best_angle = float(candidate_angles[best_index])
    best_error = float(errors[best_index])

    return best_angle, best_error


def resolve_cue_values(
    cue,
    center_frequency,
    angle=None,
    value=None,
    head_radius=8.75,
    ils=None,
):
    """
    Resolve a cue specification into explicit ITD, ILD, value, and angle.

    If angle is given:
        cue value is computed from angle.

    If value is given:
        cue is applied from value directly,
        and the closest matching angle is estimated for metadata/scoring.
    """

    cue = cue.upper()

    if angle is None and value is None:
        raise ValueError("Each stimulus must define either angle or value.")

    if angle is not None and value is not None:
        raise ValueError(
            "Define either angle or value, not both. "
            "Otherwise the cue definition is ambiguous."
        )

    if cue in ["ILD", "COMBINED"] and ils is None:
        ils = get_interaural_level_spectrum()

    cue_definition = "angle" if angle is not None else "value"

    # If value is given directly, estimate the best-matching angle.
    angle_estimation_error = None

    if angle is None and value is not None:
        angle, angle_estimation_error = value_to_angle(
            cue=cue,
            value=value,
            center_frequency=center_frequency,
            head_radius=head_radius,
            ils=ils,
        )

    resolved = {
        "cue": cue,
        "angle": angle,
        "value": value,
        "center_frequency": center_frequency,
        "head_radius": head_radius,
        "ITD": 0,
        "ILD": 0,
        "cue_definition": cue_definition,
        "angle_estimation_error": angle_estimation_error,
    }

    if cue == "ITD":

        if cue_definition == "angle":
            itd = slab.Binaural.azimuth_to_itd(
                azimuth=angle,
                frequency=center_frequency,
                head_radius=head_radius,
            )
        else:
            itd = value

        resolved["ITD"] = float(itd)
        resolved["value"] = float(itd)

    elif cue == "ILD":

        if cue_definition == "angle":
            ild = slab.Binaural.azimuth_to_ild(
                azimuth=angle,
                frequency=center_frequency,
                ils=ils,
            )
            ild_scalar = ild_tuple_to_scalar(ild)
        else:
            ild_scalar = value

        resolved["ILD"] = float(ild_scalar)
        resolved["value"] = float(ild_scalar)

    elif cue == "COMBINED":

        if cue_definition == "angle":
            itd = slab.Binaural.azimuth_to_itd(
                azimuth=angle,
                frequency=center_frequency,
                head_radius=head_radius,
            )

            ild = slab.Binaural.azimuth_to_ild(
                azimuth=angle,
                frequency=center_frequency,
                ils=ils,
            )
            ild = ild_tuple_to_scalar(ild)

        else:
            if not isinstance(value, dict):
                raise TypeError(
                    "For cue='COMBINED' with direct values, value must be a dict, "
                    "for example {'ITD': 0.00018, 'ILD': 4.5}."
                )

            itd = value.get("ITD", 0)
            ild = value.get("ILD", 0)

        resolved["ITD"] = float(itd)
        resolved["ILD"] = float(ild)
        resolved["value"] = {
            "ITD": float(itd),
            "ILD": float(ild),
        }

    else:
        raise ValueError(
            f"Unknown cue '{cue}'. Use 'ITD', 'ILD', or 'COMBINED'."
        )

    return resolved


def angle_to_cue_value(
    cue,
    angle,
    center_frequency,
    head_radius=8.75,
    ils=None,
):
    """
    Convert an azimuth angle into the cue value used by a given cue.

    For ITD:
        returns scalar ITD

    For ILD:
        returns scalar ILD

    For COMBINED:
        returns {"ITD": ..., "ILD": ...}
    """

    resolved = resolve_cue_values(
        cue=cue,
        center_frequency=center_frequency,
        angle=angle,
        value=None,
        head_radius=head_radius,
        ils=ils,
    )

    return resolved["value"]


def apply_cue(
    sound,
    cue,
    center_frequency,
    angle=None,
    value=None,
    head_radius=8.75,
    ils=None,
):
    """
    Apply ITD, ILD, or COMBINED cue to a sound.

    Cues can be defined either by angle or by direct cue value.
    If value is given, an equivalent angle is estimated for metadata.
    """

    binaural = ensure_binaural(sound)

    resolved = resolve_cue_values(
        cue=cue,
        center_frequency=center_frequency,
        angle=angle,
        value=value,
        head_radius=head_radius,
        ils=ils,
    )

    if resolved["cue"] == "ITD":
        cued_sound = binaural.itd(duration=resolved["ITD"])

    elif resolved["cue"] == "ILD":
        cued_sound = binaural.ild(dB=resolved["ILD"])

    elif resolved["cue"] == "COMBINED":
        cued_sound = binaural.itd(duration=resolved["ITD"])
        cued_sound = cued_sound.ild(dB=resolved["ILD"])

    else:
        raise ValueError(f"Unknown cue: {resolved['cue']}")

    return cued_sound, resolved


def generate_stim(
    center_frequency,
    duration=0.3,
    samplerate=44100,
    level=80
):
    """
    Generate binaural noise, bandpass-filtered at 1/3 octave around
    a center frequency, then A-weighted.

    Parameters
    ----------
    center_frequency : int or float
        Center frequency of the bandpass filter in Hz.

    duration : float
        Duration of the sound in seconds.

    samplerate : int
        Sampling rate in Hz.

    level : int or float
        Target sound level in dB after filtering and A-weighting.

    kind : str
        Binaural noise type. Usually "diotic" for identical noise
        in both ears before applying ITD/ILD cues.

    Returns
    -------
    sound : slab.Binaural
        Binaural, bandpass-filtered, A-weighted noise.

    info : dict
        Dictionary with stimulus parameters.
    """

    low_cutoff = center_frequency / 2 ** (1 / 6)
    high_cutoff = center_frequency * 2 ** (1 / 6)

    if high_cutoff >= samplerate / 2:
        raise ValueError(
            f"Upper cutoff frequency ({high_cutoff:.1f} Hz) exceeds "
            f"Nyquist frequency ({samplerate / 2:.1f} Hz)."
        )

    # Start with diotic binaural noise
    sound = slab.Binaural.whitenoise(
        duration=duration,
        samplerate=samplerate
    )

    # 1/3-octave bandpass filter
    sound = sound.filter(
        frequency=(low_cutoff, high_cutoff),
        kind="bp",
    )

    # Ramp stimulus to avoid clicks
    sound = sound.ramp(duration=0.01)

    # Set level
    sound.level = level

    # A-weighting
    level_aweight = sound.aweight().level
    level_diff = sound.level - level_aweight
    sound.level = sound.level + level_diff
    sound = sound.aweight()
    sound.level = sound.level - 3

    info = {
        "center_frequency": center_frequency,
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
        "bandwidth_octaves": 1 / 3,
        "duration": duration,
        "samplerate": samplerate,
        "level": level,
        "a_weighted": True,
    }

    return sound, info
