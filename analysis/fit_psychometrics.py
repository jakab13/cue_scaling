from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import psignifit as ps
import psignifit.psigniplot as psp

from stimuli.sound_handler import angle_to_cue_value


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def safe_name(value):
    """
    Convert a string/value into a file-system-safe name.
    """
    value = str(value)
    value = value.replace("-->", "_to_")
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value.strip("_")


def get_value_unit(cue):
    """
    Return the physical unit of a cue value.
    """
    cue = str(cue).upper()

    if cue == "ILD":
        return "dB"

    if cue == "ITD":
        return "s"

    if cue == "COMBINED":
        return "ITD_s;ILD_dB"

    return "unknown"


def format_number(value):
    """
    Format numbers compactly for filenames.

    Examples
    --------
    15.0 -> "15"
    2.5  -> "2p5"
    """
    if value is None:
        return "NA"

    value = float(value)

    if value.is_integer():
        return str(int(value))

    return str(value).replace(".", "p").replace("-", "m")



def make_fit_name(group_info):
    """
    Create a human-readable filename stem for one psychometric fit.

    Examples
    --------
    itd-to-ild_1400Hz_ref15deg
    itd-to-ild_500to1400Hz_ref15deg
    """

    reference_cue = str(group_info.get("reference_cue", "ref")).lower()
    comparison_cue = str(group_info.get("comparison_cue", "cmp")).lower()

    reference_freq = format_number(group_info.get("reference_center_frequency"))
    comparison_freq = format_number(
        group_info.get(
            "comparison_center_frequency",
            group_info.get("reference_center_frequency"),
        )
    )

    reference_angle = format_number(group_info.get("reference_angle_abs"))

    cue_pair = f"{reference_cue}-to-{comparison_cue}"

    if reference_freq == comparison_freq:
        freq_label = f"{reference_freq}Hz"
    else:
        freq_label = f"{reference_freq}to{comparison_freq}Hz"

    ref_label = f"ref{reference_angle}deg"

    return safe_name(f"{cue_pair}_{freq_label}_{ref_label}")


def make_psychometric_paths(root, group_info):
    """
    Create paths for the fit JSON and diagnostic figure.

    Structure:
        root/fits/subject_id/<fit_name>.json
        root/figures/subject_id/<fit_name>.png
    """

    root = Path(root)

    subject_id = safe_name(group_info.get("subject_id", "unknown_subject"))
    fit_name = make_fit_name(group_info)

    fit_dir = root / "fits" / subject_id
    figure_dir = root / "figures" / subject_id

    fit_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    json_path = fit_dir / f"{fit_name}.json"
    fig_path = figure_dir / f"{fit_name}.png"

    return json_path, fig_path, fit_name


def make_hash(payload):
    """
    Create a stable hash from dictionaries/lists/numpy-like data.
    """
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:12]


def convert_fit_angles_to_values(
    fit_summary,
    group_info,
    head_radius=8.75,
):
    """
    Convert angle-based fit parameters into comparison-cue signal values.

    The psychometric fit is performed in comparison-angle coordinates.
    This function adds equivalent cue-value parameters for the comparison cue.

    For ILD:
        angle -> dB

    For ITD:
        angle -> seconds

    For COMBINED:
        angle -> dict with ITD and ILD values
    """

    comparison_cue = group_info.get("comparison_cue")
    comparison_frequency = group_info.get("comparison_center_frequency")

    pse_angle = fit_summary.get("PSE_angle", fit_summary.get("PSE"))
    threshold_84_angle = fit_summary.get(
        "threshold_84_angle",
        fit_summary.get("threshold_84"),
    )

    if pse_angle is None or np.isnan(pse_angle):
        return {
            "PSE_value": np.nan,
            "threshold_84_value": np.nan,
            "JND_84_value": np.nan,
            "PSE_value_unit": get_value_unit(comparison_cue),
        }

    if threshold_84_angle is None or np.isnan(threshold_84_angle):
        threshold_84_value = np.nan
    else:
        threshold_84_value = angle_to_cue_value(
            cue=comparison_cue,
            angle=threshold_84_angle,
            center_frequency=comparison_frequency,
            head_radius=head_radius,
        )

    pse_value = angle_to_cue_value(
        cue=comparison_cue,
        angle=pse_angle,
        center_frequency=comparison_frequency,
        head_radius=head_radius,
    )

    # For scalar cues, compute value-space JND directly.
    if isinstance(pse_value, dict):
        value_summary = {}

        for key in pse_value:
            value_summary[f"PSE_{key}_value"] = pse_value[key]

            if isinstance(threshold_84_value, dict):
                value_summary[f"threshold_84_{key}_value"] = threshold_84_value[key]
                value_summary[f"JND_84_{key}_value"] = (
                    threshold_84_value[key] - pse_value[key]
                )
            else:
                value_summary[f"threshold_84_{key}_value"] = np.nan
                value_summary[f"JND_84_{key}_value"] = np.nan

        value_summary["PSE_value_unit"] = get_value_unit(comparison_cue)
        return value_summary

    else:
        jnd_84_value = threshold_84_value - pse_value

        return {
            "PSE_value": pse_value,
            "threshold_84_value": threshold_84_value,
            "JND_84_value": jnd_84_value,
            "PSE_value_unit": get_value_unit(comparison_cue),
        }


def load_run_csv(path):
    """
    Load one recorded run.
    """
    df = pd.read_csv(path)

    if "is_correct" not in df.columns:
        raise ValueError("CSV must contain an 'is_correct' column.")

    # Convert possible string booleans to real booleans
    if df["is_correct"].dtype == object:
        df["is_correct"] = df["is_correct"].astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False}
        )

    return df


def load_run_files(csv_paths):
    """
    Load and concatenate multiple run CSV files.
    Adds a source_file column so each trial can still be traced back.
    """

    dfs = []

    for path in csv_paths:
        path = Path(path)
        df = load_run_csv(path)
        df["source_file"] = str(path)
        dfs.append(df)

    if not dfs:
        raise ValueError("No CSV files provided.")

    return pd.concat(dfs, ignore_index=True)


def add_folded_coordinates(
    df,
    reference_angle_col="reference_angle",
    comparison_angle_col="comparison_angle",
):
    """
    Add folded reference/comparison angles.

    Mirrored trials are folded around the midline, assuming no left-right bias.

    Example:
        reference_angle = -8, comparison_angle = -20
        folded_reference_angle = 8
        folded_comparison_angle = 20
    """

    df = df.copy()

    if reference_angle_col not in df.columns:
        raise ValueError(f"Missing column: {reference_angle_col}")

    if comparison_angle_col not in df.columns:
        raise ValueError(f"Missing column: {comparison_angle_col}")

    reference_angle = df[reference_angle_col].astype(float)
    comparison_angle = df[comparison_angle_col].astype(float)

    # Use reference sign for folding.
    # If reference is exactly 0, default to +1.
    fold_sign = np.sign(reference_angle)
    fold_sign = fold_sign.replace(0, 1)

    df["reference_angle_abs"] = reference_angle.abs()
    df["comparison_angle_folded"] = comparison_angle * fold_sign

    return df


def add_comparison_right_response(df):
    """
    Add binary response columns for psychometric fitting.

    comparison_right:
        1 = participant judged comparison further right than reference
            in physical signed coordinates.

    comparison_right_folded:
        same response, but folded into the positive-reference coordinate system.
        This is the variable to use when fitting against comparison_angle_folded.
    """

    df = df.copy()

    required_cols = ["response", "first", "second", "reference_angle"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns needed for response recoding: {missing}")

    def recode(row):
        response = str(row["response"]).lower()
        first = row["first"]
        second = row["second"]

        if response not in ["left", "right"]:
            return None

        # If participant says "right", they mean second > first.
        # If participant says "left", they mean second < first.
        if response == "right":
            judged_rightward = second
            judged_leftward = first
        else:
            judged_rightward = first
            judged_leftward = second

        if judged_rightward == "comparison" and judged_leftward == "reference":
            return 1

        if judged_rightward == "reference" and judged_leftward == "comparison":
            return 0

        return None

    df["comparison_right"] = df.apply(recode, axis=1)

    # Fold responses whenever the reference was presented on the negative side.
    # This matches the folding of comparison angles.
    negative_reference = df["reference_angle"].astype(float) < 0

    df["comparison_right_folded"] = df["comparison_right"]

    df.loc[negative_reference, "comparison_right_folded"] = (
        1 - df.loc[negative_reference, "comparison_right"]
    )

    return df


DEFAULT_GROUP_COLS = [
    "subject_id",
    "reference_cue",
    "reference_angle_abs",
    "reference_center_frequency",
    "comparison_cue",
    "comparison_center_frequency",
]


def get_available_group_cols(df, group_cols=None):
    """
    Use only grouping columns that exist in the dataframe.
    """
    if group_cols is None:
        group_cols = DEFAULT_GROUP_COLS

    return [col for col in group_cols if col in df.columns]


def iter_psychometric_groups(df, group_cols=None):
    """
    Yield grouped psychometric datasets.
    """

    group_cols = get_available_group_cols(df, group_cols)

    if not group_cols:
        raise ValueError("No valid grouping columns found.")

    for group_key, group_df in df.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_info = dict(zip(group_cols, group_key))

        yield group_info, group_df.copy()


def make_psignifit_data(
    group_df,
    x_col="comparison_angle_folded",
    response_col="comparison_right_folded",
    bin_col="comparison_index",
):
    """
    Convert trial-level data into psignifit's n x 3 format:

        stimulus level | n comparison-right responses | n total

    Trials are grouped by comparison_index when available. This avoids
    treating mirrored/non-mirrored trials as different levels due to tiny
    floating-point differences in comparison_angle_folded.
    """

    if x_col not in group_df.columns:
        raise ValueError(f"Missing x column: {x_col}")

    if response_col not in group_df.columns:
        raise ValueError(f"Missing response column: {response_col}")

    group_df = group_df.dropna(subset=[response_col, x_col]).copy()

    if bin_col is not None and bin_col in group_df.columns:
        grouped = (
            group_df
            .groupby(bin_col, dropna=False)
            .agg(
                x_value=(x_col, "mean"),
                n_comparison_right_folded=(response_col, "sum"),
                n_total=(response_col, "count"),
            )
            .reset_index()
            .sort_values("x_value")
        )

        data = grouped[
            ["x_value", "n_comparison_right_folded", "n_total"]
        ].to_numpy(dtype=float)

    else:
        # Fallback: round x values before grouping to avoid float issues
        group_df["x_value_rounded"] = group_df[x_col].round(3)

        grouped = (
            group_df
            .groupby("x_value_rounded", dropna=False)[response_col]
            .agg(["sum", "count"])
            .reset_index()
            .rename(
                columns={
                    "x_value_rounded": "x_value",
                    "sum": "n_comparison_right_folded",
                    "count": "n_total",
                }
            )
            .sort_values("x_value")
        )

        data = grouped[
            ["x_value", "n_comparison_right_folded", "n_total"]
        ].to_numpy(dtype=float)

    return data, grouped


def get_cached_fit_info(summary_path, subject_id, fit_name, data_hash):
    """
    Check whether a valid cached fit exists in psychometric_summary.csv.

    Returns the matching summary row if:
        - subject_id matches
        - fit_name matches
        - data_hash matches
        - fit_json exists

    Otherwise returns None.
    """

    summary_path = Path(summary_path)

    if not summary_path.exists():
        return None

    summary = pd.read_csv(summary_path)

    required_cols = {"subject_id", "fit_name", "data_hash", "fit_json"}

    if not required_cols.issubset(summary.columns):
        return None

    matches = summary[
        (summary["subject_id"].astype(str) == str(subject_id))
        & (summary["fit_name"].astype(str) == str(fit_name))
        & (summary["data_hash"].astype(str) == str(data_hash))
    ]

    if matches.empty:
        return None

    row = matches.iloc[-1]

    fit_json = Path(row["fit_json"])

    if not fit_json.exists():
        return None

    return row


def fit_or_load_psychometric(
    data,
    group_info,
    json_path,
    summary_path,
    fit_name,
    sigmoid="norm",
    experiment_type="yes/no",
    overwrite=False,
):
    """
    Fit one psychometric function or load an existing cached fit.

    Cache identity is stored in psychometric_summary.csv using:
        subject_id + fit_name + data_hash

    The psignifit JSON filename remains human-readable.
    """

    fit_options = {
        "sigmoid": sigmoid,
        "experiment_type": experiment_type,
    }

    cache_payload = {
        "data": data.tolist(),
        "group_info": group_info,
        "fit_options": fit_options,
    }

    data_hash = make_hash(cache_payload)

    subject_id = group_info.get("subject_id", "unknown_subject")

    if not overwrite:
        cached = get_cached_fit_info(
            summary_path=summary_path,
            subject_id=subject_id,
            fit_name=fit_name,
            data_hash=data_hash,
        )

        if cached is not None:
            result = ps.Result.load_json(cached["fit_json"])
            return result, data_hash, True

    result = ps.psignifit(
        data,
        sigmoid=sigmoid,
        experiment_type=experiment_type,
    )

    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_json(json_path)

    return result, data_hash, False

def extract_fit_summary(result):
    """
    Extract key parameters from a psignifit Result object.
    """

    params = result.parameter_estimate
    cis = result.confidence_intervals

    pse = params.get("threshold", np.nan)
    width = params.get("width", np.nan)
    lapse = params.get("lambda", np.nan)
    gamma = params.get("gamma", np.nan)
    eta = params.get("eta", np.nan)

    # 50% unscaled threshold equals the fitted threshold definition.
    pse_50, pse_50_ci = result.threshold(0.5, unscaled=True)

    # 84% correct point on the scaled psychometric function.
    threshold_84, threshold_84_ci = result.threshold(0.84, unscaled=True)

    jnd_84 = threshold_84 - pse_50

    summary = {
        "PSE_angle": pse_50,
        "JND_84_angle": jnd_84,
        "threshold_84_angle": threshold_84,
        "width": width,
        "lambda": lapse,
        "gamma": gamma,
        "eta": eta,
    }

    # Add common confidence intervals where available
    for param_name in ["threshold", "width", "lambda", "gamma", "eta"]:
        if param_name in cis:
            for ci_level, bounds in cis[param_name].items():
                summary[f"{param_name}_ci{ci_level}_low"] = bounds[0]
                summary[f"{param_name}_ci{ci_level}_high"] = bounds[1]

    for ci_level, bounds in threshold_84_ci.items():
        summary[f"threshold_84_ci{ci_level}_low"] = bounds[0]
        summary[f"threshold_84_ci{ci_level}_high"] = bounds[1]

    # Approximate JND interval from the two threshold intervals.
    # Conservative, but useful for screening.
    for ci_level in threshold_84_ci:
        if ci_level in pse_50_ci:
            summary[f"JND_84_ci{ci_level}_low"] = (
                threshold_84_ci[ci_level][0] - pse_50_ci[ci_level][1]
            )
            summary[f"JND_84_ci{ci_level}_high"] = (
                threshold_84_ci[ci_level][1] - pse_50_ci[ci_level][0]
            )

    return summary


def save_psychometric_figure(
    result,
    fig_path,
    title=None,
    x_label="Folded comparison angle (deg)",
    reference_x=None,
):
    """
    Save a psychometric fit figure.

    Optionally adds a vertical line showing the reference/reference location.
    """

    fig, ax = plt.subplots(figsize=(5, 4))

    psp.plot_psychometric_function(
        result,
        ax=ax,
        x_label=x_label,
        y_label="p('comparison further right')",
    )

    if reference_x is not None:
        ax.vlines(
            x=reference_x,
            ymin=0,
            ymax=0.5,
            linestyles="--",
            linewidth=1.5,
            alpha=0.5,
            label="ref"
        )

        ax.legend(loc="upper left")

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    # plt.show(block=False)
    plt.close(fig)


def fit_dataframe(
    df,
    derivatives_root="data/psychometrics",
    group_cols=None,
    x_col="comparison_angle_folded",
    sigmoid="norm",
    experiment_type="yes/no",
    overwrite=False,
    analysis_label="combined",
):
    """
    Fit psychometric functions from a trial-level dataframe.

    This can contain trials from one run or many runs.
    """

    derivatives_root = Path(derivatives_root)

    summary_path = derivatives_root / "psychometric_summary.csv"

    if "comparison_angle_folded" not in df.columns:
        if "comparison_angle" in df.columns:
            df = add_folded_coordinates(
                df,
                reference_angle_col="reference_angle",
                comparison_angle_col="comparison_angle",
            )
        elif "comparison_design_angle" in df.columns:
            df = add_folded_coordinates(
                df,
                reference_angle_col="reference_angle",
                comparison_angle_col="comparison_design_angle",
            )
        else:
            raise ValueError(
                "Could not find comparison angle column. Expected "
                "'comparison_angle' or 'comparison_design_angle'."
            )

    df = add_comparison_right_response(df)

    summary_rows = []

    for group_info, group_df in iter_psychometric_groups(df, group_cols):

        data, binned = make_psignifit_data(
            group_df,
            x_col=x_col,
            response_col="comparison_right_folded",
            bin_col="comparison_index",
        )

        if len(data) < 3:
            print(f"Skipping group with <3 stimulus levels: {group_info}")
            continue

        json_path, fig_path, fit_name = make_psychometric_paths(
            root=derivatives_root,
            group_info=group_info,
        )

        result, data_hash, loaded = fit_or_load_psychometric(
            data=data,
            group_info=group_info,
            json_path=json_path,
            summary_path=summary_path,
            fit_name=fit_name,
            sigmoid=sigmoid,
            experiment_type=experiment_type,
            overwrite=overwrite,
        )

        title = (
            f"{group_info.get('reference_cue', '')} reference "
            f"{group_info.get('reference_angle_abs', '')}° → "
            f"{group_info.get('comparison_cue', '')}\n"
            f"{group_info.get('reference_center_frequency', '')} Hz"
        )

        if overwrite or not fig_path.exists() or not loaded:
            save_psychometric_figure(
                result=result,
                fig_path=fig_path,
                title=title,
                x_label="Comparison angle (deg)",
                reference_x=group_info.get("reference_angle_abs"),
            )

        fit_summary = extract_fit_summary(result)

        value_summary = convert_fit_angles_to_values(
            fit_summary=fit_summary,
            group_info=group_info,
        )

        source_files = (
            sorted(group_df["source_file"].unique().tolist())
            if "source_file" in group_df.columns
            else []
        )

        row = {
            "fit_name": fit_name,
            "data_hash": data_hash,
            "analysis_label": analysis_label,
            "fit_json": str(json_path),
            "fit_figure": str(fig_path),
            "fit_loaded_from_cache": loaded,
            "n_trials": int(data[:, 2].sum()),
            "n_levels": int(len(data)),
            "n_source_files": len(source_files),
            "source_files": ";".join(source_files),
            **group_info,
            **fit_summary,
            **value_summary,
        }

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = derivatives_root / "psychometric_summary.csv"
    derivatives_root.mkdir(parents=True, exist_ok=True)

    if summary_path.exists() and not overwrite:
        old = pd.read_csv(summary_path)
        combined = pd.concat([old, summary_df], ignore_index=True)

        combined = combined.drop_duplicates(
            subset=["subject_id", "fit_name"],
            keep="last",
        )
    else:
        combined = summary_df

    combined.to_csv(summary_path, index=False)

    return summary_df


def fit_run_file(
    csv_path,
    derivatives_root="data/psychometrics",
    group_cols=None,
    x_col="comparison_angle_folded",
    sigmoid="norm",
    experiment_type="yes/no",
    overwrite=False,
):
    """
    Fit psychometric functions from one run CSV.
    """

    csv_path = Path(csv_path)
    df = load_run_files([csv_path])

    return fit_dataframe(
        df=df,
        derivatives_root=derivatives_root,
        group_cols=group_cols,
        x_col=x_col,
        sigmoid=sigmoid,
        experiment_type=experiment_type,
        overwrite=overwrite,
        analysis_label=csv_path.stem,
    )


def fit_run_files(
    csv_paths,
    derivatives_root="data/psychometrics",
    group_cols=None,
    x_col="comparison_angle_folded",
    sigmoid="norm",
    experiment_type="yes/no",
    overwrite=False,
    analysis_label="combined_runs",
):
    """
    Fit psychometric functions from multiple run CSV files.
    Trials belonging to the same psychometric group are pooled.
    """

    df = load_run_files(csv_paths)

    return fit_dataframe(
        df=df,
        derivatives_root=derivatives_root,
        group_cols=group_cols,
        x_col=x_col,
        sigmoid=sigmoid,
        experiment_type=experiment_type,
        overwrite=overwrite,
        analysis_label=analysis_label,
    )

# if __name__ == "__main__":
#
#     summary = fit_run_file(
#         csv_path="data/raw/jakab/jakab_run_2026-05-21_17-18-36.csv",
#         derivatives_root="data/psychometrics",
#         sigmoid="norm",
#         experiment_type="yes/no",
#         overwrite=False,
#     )
#
#     print(summary[[
#         "condition_id",
#         "reference_cue",
#         "reference_angle_abs",
#         "comparison_cue",
#         "PSE",
#         "JND_84",
#         "eta",
#         "fit_loaded_from_cache",
#     ]])

