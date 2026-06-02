from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import psignifit as ps
import psignifit.psigniplot as psp

CUE_COLORS = {
    "ILD": "C0",       # matplotlib default blue
    "ITD": "C1",       # matplotlib default orange
    "COMBINED": "C2",  # matplotlib default green
}


def get_cue_color(cue):
    """
    Return project-wide colour for cue type.
    """

    cue = str(cue).upper()
    return CUE_COLORS.get(cue, None)


def is_jnd_condition(group_info):
    """
    Identify centred within-cue JND conditions.

    JND condition:
        reference_angle_folded == 0
        reference_cue == comparison_cue
    """

    reference_cue = str(group_info.get("reference_cue", "")).upper()
    comparison_cue = str(group_info.get("comparison_cue", "")).upper()

    reference_angle = float(group_info.get("reference_angle_folded", np.nan))

    return (
        np.isfinite(reference_angle)
        and reference_angle == 0
        and reference_cue == comparison_cue
    )

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



def make_fit_name(group_info, fit_domain):
    """
    Create a human-readable filename stem for one psychometric fit.

    Examples
    --------
    ITD-to-ILD_400Hz_ref5deg_angle
    ITD-to-ILD_400Hz_ref5deg_value
    """

    reference_cue = str(group_info.get("reference_cue", "REF")).upper()
    comparison_cue = str(group_info.get("comparison_cue", "CMP")).upper()

    reference_freq = format_number(group_info.get("reference_center_frequency"))
    comparison_freq = format_number(
        group_info.get(
            "comparison_center_frequency",
            group_info.get("reference_center_frequency"),
        )
    )

    reference_angle = format_number(group_info.get("reference_angle_folded"))

    cue_pair = f"{reference_cue}-to-{comparison_cue}"

    if reference_freq == comparison_freq:
        freq_label = f"{reference_freq}Hz"
    else:
        freq_label = f"{reference_freq}to{comparison_freq}Hz"

    ref_label = f"ref{reference_angle}deg"

    return safe_name(f"{cue_pair}_{freq_label}_{ref_label}_{fit_domain}")


def make_frequency_folder_name(group_info):
    """
    Create frequency subfolder name for psychometric outputs.

    Examples
    --------
    400Hz
    500to1400Hz
    """

    reference_freq = format_number(group_info.get("reference_center_frequency"))
    comparison_freq = format_number(
        group_info.get(
            "comparison_center_frequency",
            group_info.get("reference_center_frequency"),
        )
    )

    if reference_freq == comparison_freq:
        freq_label = f"{reference_freq}Hz"
    else:
        freq_label = f"{reference_freq}to{comparison_freq}Hz"

    return safe_name(freq_label)


def make_psychometric_paths(root, group_info, fit_domain):
    """
    Create paths for the fit JSON and diagnostic figure.

    Structure:
        root/fits/subject_id/fit_domain/frequency/<fit_name>.json
        root/figures/subject_id/fit_domain/frequency/<fit_name>.png

    Example:
        data/psychometrics/fits/jakab/angle/400Hz/ITD-to-ILD_400Hz_ref5deg_angle.json
        data/psychometrics/fits/jakab/value/400Hz/ITD-to-ILD_400Hz_ref5deg_value.json
    """

    root = Path(root)

    subject_id = safe_name(group_info.get("subject_id", "unknown_subject"))
    frequency_folder = make_frequency_folder_name(group_info)
    fit_domain = safe_name(fit_domain)

    fit_name = make_fit_name(
        group_info=group_info,
        fit_domain=fit_domain,
    )

    fit_dir = root / "fits" / subject_id / fit_domain / frequency_folder
    figure_dir = root / "figures" / subject_id / fit_domain / frequency_folder

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


def print_fit_status(group_info):
    """
    Print one compact status line for the current psychometric fit.
    """

    subject_id = group_info.get("subject_id", "")
    reference_cue = group_info.get("reference_cue", "")
    comparison_cue = group_info.get("comparison_cue", "")
    frequency = group_info.get("reference_center_frequency", "")
    reference_angle = group_info.get("reference_angle_folded", "")

    print(
        f"Calculating fit... "
        f"{subject_id} | "
        f"{reference_cue}→{comparison_cue} | "
        f"{frequency} Hz | "
        f"ref {reference_angle}°",
        flush=True,
    )

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
    "reference_angle_folded",
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

    print_fit_status(group_info)

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

    The fitted x-axis can be angle or value.
    """

    params = result.parameter_estimate
    cis = result.confidence_intervals

    width = params.get("width", np.nan)
    lapse = params.get("lambda", np.nan)
    gamma = params.get("gamma", np.nan)
    eta = params.get("eta", np.nan)

    pse, pse_ci = result.threshold(0.5, unscaled=True)
    threshold_84, threshold_84_ci = result.threshold(0.84, unscaled=True)

    jnd_84 = threshold_84 - pse

    summary = {
        "PSE": pse,
        "JND_84": jnd_84,
        "threshold_84": threshold_84,
        "width": width,
        "lambda": lapse,
        "gamma": gamma,
        "eta": eta,
    }

    for param_name in ["threshold", "width", "lambda", "gamma", "eta"]:
        if param_name in cis:
            for ci_level, bounds in cis[param_name].items():
                summary[f"{param_name}_ci{ci_level}_low"] = bounds[0]
                summary[f"{param_name}_ci{ci_level}_high"] = bounds[1]

    for ci_level, bounds in threshold_84_ci.items():
        summary[f"threshold_84_ci{ci_level}_low"] = bounds[0]
        summary[f"threshold_84_ci{ci_level}_high"] = bounds[1]

    for ci_level in threshold_84_ci:
        if ci_level in pse_ci:
            summary[f"JND_84_ci{ci_level}_low"] = (
                threshold_84_ci[ci_level][0] - pse_ci[ci_level][1]
            )
            summary[f"JND_84_ci{ci_level}_high"] = (
                threshold_84_ci[ci_level][1] - pse_ci[ci_level][0]
            )

    return summary


def get_fit_specs_for_group(
    group_info,
    group_df,
    requested_fit_domains=("angle", "value"),
    angle_x_col="comparison_angle_folded",
    value_x_col="comparison_value_folded",
):
    """
    Decide which psychometric fits to run for one group.

    By default:
        - angle fit is run when comparison_angle_folded exists
        - value fit is run when comparison_value_folded exists
        - value fit is skipped when comparison_cue == COMBINED
    """

    specs = []

    requested_fit_domains = tuple(requested_fit_domains)

    comparison_cue = str(group_info.get("comparison_cue", "")).upper()

    if "angle" in requested_fit_domains:
        if angle_x_col in group_df.columns:
            specs.append(
                {
                    "fit_domain": "angle",
                    "x_col": angle_x_col,
                    "x_unit": "deg",
                    "x_label": "Comparison angle (deg)",
                }
            )

    if "value" in requested_fit_domains:
        if comparison_cue != "COMBINED" and value_x_col in group_df.columns:

            specs.append(
                {
                    "fit_domain": "value",
                    "x_col": value_x_col,
                    "x_unit": get_value_unit(comparison_cue),
                    "x_label": (
                        f"Comparison {comparison_cue} "
                        f"({get_value_unit(comparison_cue)})"
                    ),
                }
            )

    return specs


def save_psychometric_figure(
    result,
    fig_path,
    title=None,
    x_label="Folded comparison angle (deg)",
    reference_x=None,
    color='#000000',
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
        data_color=color,
        line_color=color,
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
    plt.close(fig)


def filter_main_analysis_runs(df, include_practice=False):
    """
    Exclude practice/test/pilot runs unless explicitly requested.
    """

    if include_practice:
        return df

    if "include_in_analysis" not in df.columns:
        return df

    include = df["include_in_analysis"]

    if include.dtype == bool:
        return df[include]

    include = include.astype(str).str.lower().isin(["true", "1", "yes"])

    return df[include]



def fit_dataframe(
    df,
    derivatives_root="data/psychometrics",
    group_cols=None,
    x_col="comparison_angle_folded",
    fit_domains=("angle", "value"),
    sigmoid="norm",
    experiment_type="yes/no",
    overwrite=False,
    analysis_label="combined",
    include_practice=False,
):
    """
    Fit psychometric functions from a trial-level dataframe.

    This can contain trials from one run or many runs.
    """

    derivatives_root = Path(derivatives_root)

    summary_path = derivatives_root / "psychometric_summary.csv"

    df = filter_main_analysis_runs(df, include_practice=include_practice)

    if not include_practice and "include_in_analysis" in df.columns:
        df = df[df["include_in_analysis"].astype(str).str.lower().isin(["true", "1"])]

    df = add_comparison_right_response(df)

    summary_rows = []

    for group_info, group_df in iter_psychometric_groups(df, group_cols):

        fit_specs = get_fit_specs_for_group(
            group_info=group_info,
            group_df=group_df,
            requested_fit_domains=fit_domains,
            angle_x_col=x_col,
            value_x_col="comparison_value_folded",
        )

        if not fit_specs:
            print(f"No valid fit domains for group: {group_info}")
            continue

        for fit_spec in fit_specs:

            fit_domain = fit_spec["fit_domain"]
            domain_x_col = fit_spec["x_col"]
            x_unit = fit_spec["x_unit"]
            x_label = fit_spec["x_label"]

            data, binned = make_psignifit_data(
                group_df,
                x_col=domain_x_col,
                response_col="comparison_right_folded",
                bin_col="comparison_index",
            )

            if len(data) < 3:
                print(
                    f"Skipping {fit_domain} fit with <3 stimulus levels: "
                    f"{group_info}"
                )
                continue

            json_path, fig_path, fit_name = make_psychometric_paths(
                root=derivatives_root,
                group_info=group_info,
                fit_domain=fit_domain,
            )

            cache_group_info = {
                **group_info,
                "fit_domain": fit_domain,
                "x_variable": domain_x_col,
                "x_unit": x_unit,
            }

            result, data_hash, loaded = fit_or_load_psychometric(
                data=data,
                group_info=cache_group_info,
                json_path=json_path,
                summary_path=summary_path,
                fit_name=fit_name,
                sigmoid=sigmoid,
                experiment_type=experiment_type,
                overwrite=overwrite,
            )

            title = (
                f"{group_info.get('reference_cue', '')} reference "
                f"{group_info.get('reference_angle_folded', '')}° → "
                f"{group_info.get('comparison_cue', '')}\n"
                f"{group_info.get('reference_center_frequency', '')} Hz "
                f"({fit_domain} fit)"
            )

            if fit_domain == "angle":
                reference_x = group_info.get("reference_angle_folded")
            else:
                reference_x = None

            if is_jnd_condition(group_info):
                figure_color = get_cue_color(group_info.get("comparison_cue"))
            else:
                figure_color = "#000000"

            if overwrite or not fig_path.exists() or not loaded:
                save_psychometric_figure(
                    result=result,
                    fig_path=fig_path,
                    title=title,
                    x_label=x_label,
                    reference_x=reference_x,
                    color=figure_color,
                )

            fit_summary = extract_fit_summary(result)

            domain_summary = {
                "fit_domain": fit_domain,
                "x_variable": domain_x_col,
                "x_unit": x_unit,
            }

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
                **domain_summary,
            }

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_path = derivatives_root / "psychometric_summary.csv"
    derivatives_root.mkdir(parents=True, exist_ok=True)

    if summary_path.exists() and not overwrite:
        old = pd.read_csv(summary_path)
        combined = pd.concat([old, summary_df], ignore_index=True)

        combined = combined.drop_duplicates(
            subset=["subject_id", "fit_domain", "fit_name"],
            keep="last",
        )
    else:
        combined = summary_df

    combined.to_csv(summary_path, index=False)

    print(f"Finished fitting: {summary_path}", flush=True)

    return summary_df


def fit_run_file(
    csv_path,
    derivatives_root="data/psychometrics",
    group_cols=None,
    x_col="comparison_angle_folded",
    fit_domains=("angle", "value"),
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
        fit_domains=fit_domains,
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
    fit_domains=("angle", "value"),
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
        fit_domains=fit_domains,
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
#         "reference_angle_folded",
#         "comparison_cue",
#         "PSE",
#         "JND_84",
#         "eta",
#         "fit_loaded_from_cache",
#     ]])

