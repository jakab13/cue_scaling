from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import psignifit as ps
import psignifit.psigniplot as psp


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


def make_hash(payload):
    """
    Create a stable hash from dictionaries/lists/numpy-like data.
    """
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:12]


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


def add_folded_coordinates(
    df,
    standard_angle_col="standard_angle",
    comparison_angle_col="comparison_angle",
):
    """
    Add folded standard/comparison angles.

    Mirrored trials are folded around the midline, assuming no left-right bias.

    Example:
        standard_angle = -8, comparison_angle = -20
        folded_standard_angle = 8
        folded_comparison_angle = 20
    """

    df = df.copy()

    if standard_angle_col not in df.columns:
        raise ValueError(f"Missing column: {standard_angle_col}")

    if comparison_angle_col not in df.columns:
        raise ValueError(f"Missing column: {comparison_angle_col}")

    standard_angle = df[standard_angle_col].astype(float)
    comparison_angle = df[comparison_angle_col].astype(float)

    # Use standard sign for folding.
    # If standard is exactly 0, default to +1.
    fold_sign = np.sign(standard_angle)
    fold_sign = fold_sign.replace(0, 1)

    df["standard_angle_abs"] = standard_angle.abs()
    df["comparison_angle_folded"] = comparison_angle * fold_sign

    return df


def add_comparison_right_response(df):
    """
    Add binary response columns for psychometric fitting.

    comparison_right:
        1 = participant judged comparison further right than standard
            in physical signed coordinates.

    comparison_right_folded:
        same response, but folded into the positive-standard coordinate system.
        This is the variable to use when fitting against comparison_angle_folded.
    """

    df = df.copy()

    required_cols = ["response", "first", "second", "standard_angle"]
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

        if judged_rightward == "comparison" and judged_leftward == "standard":
            return 1

        if judged_rightward == "standard" and judged_leftward == "comparison":
            return 0

        return None

    df["comparison_right"] = df.apply(recode, axis=1)

    # Fold responses whenever the standard was presented on the negative side.
    # This matches the folding of comparison angles.
    negative_standard = df["standard_angle"].astype(float) < 0

    df["comparison_right_folded"] = df["comparison_right"]

    df.loc[negative_standard, "comparison_right_folded"] = (
        1 - df.loc[negative_standard, "comparison_right"]
    )

    return df


DEFAULT_GROUP_COLS = [
    "subject_id",
    "condition_id",
    "standard_cue",
    "standard_angle_abs",
    "standard_center_frequency",
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


def fit_or_load_psychometric(
    data,
    group_info,
    output_dir,
    sigmoid="norm",
    experiment_type="yes/no",
    overwrite=False,
):
    """
    Fit one psychometric function or load an existing cached fit.

    The cache hash depends on:
        - grouped psignifit data
        - group metadata
        - fit options
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_options = {
        "sigmoid": sigmoid,
        "experiment_type": experiment_type,
    }

    cache_payload = {
        "data": data.tolist(),
        "group_info": group_info,
        "fit_options": fit_options,
    }

    fit_hash = make_hash(cache_payload)

    json_path = output_dir / f"fit_{fit_hash}.json"
    fig_path = output_dir / f"fit_{fit_hash}.png"

    if json_path.exists() and not overwrite:
        result = ps.Result.load_json(json_path)
        fit_was_loaded = True
    else:
        result = ps.psignifit(
            data,
            sigmoid=sigmoid,
            experiment_type=experiment_type,
        )
        result.save_json(json_path)
        fit_was_loaded = False

    return result, json_path, fig_path, fit_hash, fit_was_loaded


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
        "PSE": pse,
        "JND_84": jnd_84,
        "threshold_84": threshold_84,
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
):
    """
    Save a psychometric fit figure.
    """

    fig, ax = plt.subplots(figsize=(5, 4))

    psp.plot_psychometric_function(
        result,
        ax=ax,
        x_label=x_label,
        y_label="p('further right')",
    )

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    # plt.close(fig)


def make_group_output_dir(root, group_info):
    """
    Create a smart output folder for one psychometric group.
    """

    subject = safe_name(group_info.get("subject_id", "unknown_subject"))
    condition = safe_name(group_info.get("condition_id", "condition"))

    std_cue = safe_name(group_info.get("standard_cue", "std"))
    cmp_cue = safe_name(group_info.get("comparison_cue", "cmp"))

    std_angle = group_info.get("standard_angle_abs", "angle")
    std_freq = group_info.get("standard_center_frequency", "freq")
    cmp_freq = group_info.get("comparison_center_frequency", std_freq)

    folder_name = safe_name(
        f"{condition}_{std_cue}{std_angle}deg_{cmp_cue}_{std_freq}to{cmp_freq}Hz"
    )

    return Path(root) / "fits" / subject / folder_name


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
    Fit all psychometric functions contained in one run CSV.

    Returns
    -------
    summary_df : pandas.DataFrame
        One row per psychometric fit.
    """

    csv_path = Path(csv_path)
    derivatives_root = Path(derivatives_root)

    df = load_run_csv(csv_path)

    # Adapt this column name if your current output uses comparison_design_angle.
    if "comparison_angle_folded" not in df.columns:
        if "comparison_angle" in df.columns:
            df = add_folded_coordinates(
                df,
                standard_angle_col="standard_angle",
                comparison_angle_col="comparison_angle",
            )
        elif "comparison_design_angle" in df.columns:
            df = add_folded_coordinates(
                df,
                standard_angle_col="standard_angle",
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
        )

        if len(data) < 3:
            print(f"Skipping group with <3 stimulus levels: {group_info}")
            continue

        group_dir = make_group_output_dir(
            root=derivatives_root,
            group_info=group_info,
        )

        result, json_path, fig_path, fit_hash, loaded = fit_or_load_psychometric(
            data=data,
            group_info=group_info,
            output_dir=group_dir,
            sigmoid=sigmoid,
            experiment_type=experiment_type,
            overwrite=overwrite,
        )

        title = (
            f"{group_info.get('condition_id', '')}\n"
            f"{group_info.get('standard_cue', '')} std "
            f"{group_info.get('standard_angle_abs', '')}° → "
            f"{group_info.get('comparison_cue', '')}"
        )

        if overwrite or not fig_path.exists():
            save_psychometric_figure(
                result=result,
                fig_path=fig_path,
                title=title,
                x_label="Comparison angle (deg)",
            )

        fit_summary = extract_fit_summary(result)

        row = {
            "source_file": str(csv_path),
            "fit_hash": fit_hash,
            "fit_json": str(json_path),
            "fit_figure": str(fig_path),
            "fit_loaded_from_cache": loaded,
            "n_trials": int(data[:, 2].sum()),
            "n_levels": int(len(data)),
            **group_info,
            **fit_summary,
        }

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = derivatives_root / "psychometric_summary.csv"
    derivatives_root.mkdir(parents=True, exist_ok=True)

    if summary_path.exists() and not overwrite:
        old = pd.read_csv(summary_path)
        combined = pd.concat([old, summary_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["fit_hash"], keep="last")
    else:
        combined = summary_df

    combined.to_csv(summary_path, index=False)

    return summary_df


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
#         "standard_cue",
#         "standard_angle_abs",
#         "comparison_cue",
#         "PSE",
#         "JND_84",
#         "eta",
#         "fit_loaded_from_cache",
#     ]])

