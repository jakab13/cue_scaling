# analysis/analyse_jnd_components.py

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from scipy.optimize import minimize


CUE_ORDER = ["ILD", "ITD", "COMBINED"]

CUE_COLORS = {
    "ILD": "C0",       # blue
    "ITD": "C1",      # orange
    "COMBINED": "C2", # green
}


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------

def load_psychometric_summary(derivatives_root="data/psychometrics"):
    summary_path = Path(derivatives_root) / "psychometric_summary.csv"

    if not summary_path.exists():
        raise FileNotFoundError(f"No summary table found at: {summary_path}")

    return pd.read_csv(summary_path)


def load_k_slope_summary(derivatives_root="data/psychometrics"):
    slopes_path = Path(derivatives_root) / "k_slope_summary.csv"

    if not slopes_path.exists():
        raise FileNotFoundError(
            f"No k-slope summary found at: {slopes_path}. "
            "Run estimate_k_slopes(...) first."
        )

    return pd.read_csv(slopes_path)


# ---------------------------------------------------------------------
# Extract measured JNDs
# ---------------------------------------------------------------------

def extract_measured_jnds(
    psychometric_summary,
    subject_id=None,
    cues=("ILD", "ITD", "COMBINED"),
):
    """
    Extract centred within-cue JNDs.

    These are the JND measurements where:
        fit_domain == "angle"
        reference_angle_folded == 0
        reference_cue == comparison_cue

    JND_84 is assumed to be in degrees.
    """

    df = psychometric_summary.copy()

    required_cols = [
        "subject_id",
        "fit_domain",
        "reference_cue",
        "comparison_cue",
        "reference_angle_folded",
        "reference_center_frequency",
        "JND_84",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[df["fit_domain"].astype(str).str.lower() == "angle"]

    if subject_id is not None:
        df = df[df["subject_id"].astype(str) == str(subject_id)]

    df["reference_cue"] = df["reference_cue"].astype(str).str.upper()
    df["comparison_cue"] = df["comparison_cue"].astype(str).str.upper()

    df = df[df["reference_cue"] == df["comparison_cue"]]
    df = df[df["reference_cue"].isin([cue.upper() for cue in cues])]

    df["reference_angle_folded"] = pd.to_numeric(
        df["reference_angle_folded"],
        errors="coerce",
    )

    df = df[df["reference_angle_folded"] == 0]

    df["reference_center_frequency"] = pd.to_numeric(
        df["reference_center_frequency"],
        errors="coerce",
    )

    df["JND_84"] = pd.to_numeric(df["JND_84"], errors="coerce")

    df = df.dropna(
        subset=[
            "subject_id",
            "reference_cue",
            "reference_center_frequency",
            "JND_84",
        ]
    )

    keep_cols = [
        "subject_id",
        "reference_cue",
        "reference_center_frequency",
        "JND_84",
        "fit_name",
        "fit_json",
        "fit_figure",
    ]

    keep_cols = [col for col in keep_cols if col in df.columns]

    out = df[keep_cols].copy()

    out = out.rename(
        columns={
            "reference_center_frequency": "frequency",
            "JND_84": "JND_measured",
            "fit_name": "JND_fit_name",
            "fit_json": "JND_fit_json",
            "fit_figure": "JND_fit_figure",
        }
    )

    return out


# ---------------------------------------------------------------------
# Prepare k/sigma predictors
# ---------------------------------------------------------------------

def prepare_k_sigma_predictors(
    k_slope_summary,
    subject_id=None,
    cues=("ILD", "ITD", "COMBINED"),
    comparison_cue="ILD",
):
    """
    Prepare k and sigma_y predictors.

    Expected columns in k_slope_summary:
        k_slope
        sigma_y

    k_slope:
    Mapping slope in dB / degree.

sigma_y:
    Estimated single-cue uncertainty in ILD comparison space (dB),
    derived from pairwise psychometric JNDs.

    ILD:
        sigma_y = JND_ILD->ILD / sqrt(2)

    ITD / COMBINED:
        sigma_y = sqrt(
            JND_cue->ILD^2 - sigma_ILD^2
        )
    """

    df = k_slope_summary.copy()

    required_cols = [
        "subject_id",
        "reference_cue",
        "comparison_cue",
        "reference_center_frequency",
        "comparison_center_frequency",
        "k_slope",
        "sigma_y",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in k_slope_summary: {missing}")

    if subject_id is not None:
        df = df[df["subject_id"].astype(str) == str(subject_id)]

    df["reference_cue"] = df["reference_cue"].astype(str).str.upper()
    df["comparison_cue"] = df["comparison_cue"].astype(str).str.upper()

    df = df[df["reference_cue"].isin([cue.upper() for cue in cues])]
    df = df[df["comparison_cue"] == comparison_cue.upper()]

    df["reference_center_frequency"] = pd.to_numeric(
        df["reference_center_frequency"],
        errors="coerce",
    )
    df["comparison_center_frequency"] = pd.to_numeric(
        df["comparison_center_frequency"],
        errors="coerce",
    )
    df["k_slope"] = pd.to_numeric(df["k_slope"], errors="coerce")
    df["sigma_y"] = pd.to_numeric(df["sigma_y"], errors="coerce")

    df = df.dropna(
        subset=[
            "subject_id",
            "reference_cue",
            "reference_center_frequency",
            "k_slope",
            # "sigma_y",
        ]
    )

    keep_cols = [
        "subject_id",
        "reference_cue",
        "comparison_cue",
        "reference_center_frequency",
        "comparison_center_frequency",
        "k_slope",
        "sigma_y",
        "n_points",
        "x_unit",
        "y_unit",
    ]

    keep_cols = [col for col in keep_cols if col in df.columns]

    out = df[keep_cols].copy()

    out = out.rename(
        columns={
            "reference_center_frequency": "frequency",
            "n_points": "n_k_points",
        }
    )

    return out


# ---------------------------------------------------------------------
# Main analysis table
# ---------------------------------------------------------------------

def make_jnd_component_table(
    subject_id=None,
    derivatives_root="data/psychometrics",
    cues=("ILD", "ITD", "COMBINED"),
    comparison_cue="ILD",
    save=True,
):
    """
    Build the core analysis table for the JND ~ sigma/k question.

    Output columns include:
        JND_measured
        sigma_y
        k_slope
        inv_k
        JND_predicted_sigma_over_k
        prediction_error
        prediction_ratio
    """

    derivatives_root = Path(derivatives_root)

    psychometric_summary = load_psychometric_summary(derivatives_root)
    k_slope_summary = load_k_slope_summary(derivatives_root)

    measured_jnds = extract_measured_jnds(
        psychometric_summary=psychometric_summary,
        subject_id=subject_id,
        cues=cues,
    )

    predictors = prepare_k_sigma_predictors(
        k_slope_summary=k_slope_summary,
        subject_id=subject_id,
        cues=cues,
        comparison_cue=comparison_cue,
    )

    table = measured_jnds.merge(
        predictors,
        on=["subject_id", "reference_cue", "frequency"],
        how="inner",
    )

    table = table.replace([np.inf, -np.inf], np.nan)

    table = table.dropna(
        subset=[
            "JND_measured",
            # "sigma_y",
            "k_slope",
        ]
    )

    # Derived predictors
    table["inv_k"] = 1 / table["k_slope"]
    table["JND_predicted_sigma_over_k"] = table["sigma_y"] / table["k_slope"]

    # Prediction diagnostics
    table["prediction_error"] = (
        table["JND_measured"] - table["JND_predicted_sigma_over_k"]
    )
    table["abs_prediction_error"] = table["prediction_error"].abs()

    table["prediction_ratio"] = (
        table["JND_measured"] / table["JND_predicted_sigma_over_k"]
    )

    table = table.replace([np.inf, -np.inf], np.nan)

    if save:
        out_path = derivatives_root / "jnd_component_summary.csv"

        if out_path.exists():
            old = pd.read_csv(out_path)
            combined = pd.concat([old, table], ignore_index=True)

            combined = combined.drop_duplicates(
                subset=[
                    "subject_id",
                    "reference_cue",
                    "frequency",
                ],
                keep="last",
            )
        else:
            combined = table

        combined.to_csv(out_path, index=False)
        print(f"Saved JND component summary to: {out_path}")

    return table


def load_jnd_component_summary(derivatives_root="data/psychometrics"):
    path = Path(derivatives_root) / "jnd_component_summary.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"No JND component summary found at: {path}. "
            "Run make_jnd_component_table(...) first."
        )

    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Simple model-comparison helpers
# ---------------------------------------------------------------------

def r2_score(y, y_pred):
    """
    Standard R².
    """

    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y) & np.isfinite(y_pred)
    y = y[valid]
    y_pred = y_pred[valid]

    if len(y) < 2:
        return np.nan

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - ss_res / ss_tot


def fit_simple_linear_prediction(x, y):
    """
    Fit y = a + b*x and return predictions and R².

    This is descriptive, used to compare whether JND tracks
    sigma, 1/k, or sigma/k.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return {
            "intercept": np.nan,
            "slope": np.nan,
            "r2": np.nan,
        }

    slope, intercept = np.polyfit(x, y, deg=1)
    y_pred = intercept + slope * x

    return {
        "intercept": intercept,
        "slope": slope,
        "r2": r2_score(y, y_pred),
    }


def fit_inverse_k(k, jnd):
    """
    Fit:

        JND = intercept + B / k

    and evaluate prediction error on the original JND scale.
    """

    k = np.asarray(k, dtype=float)
    jnd = np.asarray(jnd, dtype=float)

    valid = (
        np.isfinite(k)
        & np.isfinite(jnd)
        & (k > 0)
    )

    k = k[valid]
    jnd = jnd[valid]

    if len(k) < 3:
        return {
            "intercept": np.nan,
            "B": np.nan,
            "p_value": np.nan,
            "r2": np.nan,
            "n_points": len(k),
        }

    inv_k = 1 / k

    result = linregress(inv_k, jnd)

    intercept = result.intercept
    B = result.slope

    # Predictions from the inverse curve on the original k scale
    jnd_predicted = intercept + B / k

    # R² from JND prediction errors
    ss_res = np.sum((jnd - jnd_predicted) ** 2)
    ss_tot = np.sum((jnd - np.mean(jnd)) ** 2)

    r2 = 1 - ss_res / ss_tot

    return {
        "intercept": float(intercept),
        "B": float(B),
        "p_value": float(result.pvalue),
        "r2": float(r2),
        "n_points": len(k),
    }


import numpy as np
from scipy.optimize import minimize


def inverse_model(k, a, B):
    """
    JND = a + B / k
    """
    return a + B / k


def inverse_sse(params, k, jnd):
    """
    Sum of squared prediction errors.
    """
    a, B = params

    predicted = inverse_model(
        k=k,
        a=a,
        B=B,
    )

    residuals = jnd - predicted

    return np.sum(residuals ** 2)


def fit_inverse_model(k, jnd):
    """
    Directly optimise:

        JND = a + B / k

    by minimising squared prediction error.
    """

    k = np.asarray(k, dtype=float)
    jnd = np.asarray(jnd, dtype=float)

    valid = (
        np.isfinite(k)
        & np.isfinite(jnd)
        & (k > 0)
    )

    k = k[valid]
    jnd = jnd[valid]

    if len(k) < 3:
        return {
            "a": np.nan,
            "B": np.nan,
            "r2": np.nan,
            "sse": np.nan,
            "n_points": len(k),
        }

    # Reasonable starting guesses
    start = [
        np.min(jnd),       # a
        1.0,               # B
    ]

    result = minimize(
        inverse_sse,
        x0=start,
        args=(k, jnd),
        method="Nelder-Mead",
    )

    a, B = result.x

    predicted = inverse_model(
        k=k,
        a=a,
        B=B,
    )

    residuals = jnd - predicted

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum(
        (jnd - np.mean(jnd)) ** 2
    )

    r2 = 1 - ss_res / ss_tot

    return {
        "a": float(a),
        "B": float(B),
        "r2": float(r2),
        "sse": float(ss_res),
        "n_points": len(k),
        "success": result.success,
    }


def permutation_test_inverse(
    k,
    jnd,
    n_permutations=1000,
    random_state=42,
):
    """
    Test whether the inverse relationship between k and JND
    is stronger than expected under random pairing.

    Statistic:
        improvement in SSE relative to a constant-mean model.
    """

    k = np.asarray(k, dtype=float)
    jnd = np.asarray(jnd, dtype=float)

    valid = (
        np.isfinite(k)
        & np.isfinite(jnd)
        & (k > 0)
    )

    k = k[valid]
    jnd = jnd[valid]

    rng = np.random.default_rng(random_state)

    # --------------------------------
    # Null model: constant mean JND
    # --------------------------------

    null_prediction = np.mean(jnd)

    null_sse = np.sum(
        (jnd - null_prediction) ** 2
    )

    # --------------------------------
    # Observed inverse fit
    # --------------------------------

    observed_fit = fit_inverse_model(
        k,
        jnd,
    )

    observed_improvement = (
        null_sse
        - observed_fit["sse"]
    )

    # --------------------------------
    # Permutations
    # --------------------------------

    permutation_improvements = []

    for _ in range(n_permutations):

        shuffled_jnd = rng.permutation(jnd)

        fit = fit_inverse_model(
            k,
            shuffled_jnd,
        )

        shuffled_null_sse = np.sum(
            (shuffled_jnd - np.mean(shuffled_jnd)) ** 2
        )

        improvement = (
            shuffled_null_sse
            - fit["sse"]
        )

        permutation_improvements.append(
            improvement
        )

    permutation_improvements = np.asarray(
        permutation_improvements
    )

    p_value = (
        np.sum(
            permutation_improvements
            >= observed_improvement
        )
        + 1
    ) / (n_permutations + 1)

    return {
        "p_value": float(p_value),
        "observed_improvement": float(
            observed_improvement
        ),
    }


def compare_jnd_component_models(table):
    """
    Compare how well measured JND tracks:
        sigma_y
        inv_k
        sigma_y / k

    Returns one row per predictor.
    """

    predictors = {
        "sigma_y": "Sigma only",
        "inv_k": "Inverse k only",
        "JND_predicted_sigma_over_k": "Sigma / k",
    }

    rows = []

    for predictor_col, label in predictors.items():

        result = fit_simple_linear_prediction(
            x=table[predictor_col],
            y=table["JND_measured"],
        )

        rows.append(
            {
                "predictor": predictor_col,
                "label": label,
                **result,
                "n_points": len(table.dropna(subset=[predictor_col, "JND_measured"])),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_component_triptych(
    subject_id=None,
    derivatives_root="data/psychometrics",
    cues=("ILD", "ITD", "COMBINED"),
    save=False,
    save_path=None,
):
    """
    Plot the central component comparison:

        measured JND vs sigma
        measured JND vs 1/k
        measured JND vs sigma/k

    This directly tests whether JND differences track mapping slope
    more closely than sigma.
    """

    derivatives_root = Path(derivatives_root)

    table_path = derivatives_root / "jnd_component_summary.csv"

    if table_path.exists():
        table = pd.read_csv(table_path)
    else:
        table = make_jnd_component_table(
            subject_id=subject_id,
            derivatives_root=derivatives_root,
            cues=cues,
            save=True,
        )

    if subject_id is not None:
        table = table[table["subject_id"].astype(str) == str(subject_id)]

    table["reference_cue"] = table["reference_cue"].astype(str).str.upper()
    table = table[table["reference_cue"].isin([cue.upper() for cue in cues])]

    if table.empty:
        raise ValueError("No JND component data found.")

    plot_specs = [
        {
            "x": "sigma_y",
            "title": "Sigma only",
            "xlabel": "Sigma (dB)",
        },
        {
            "x": "inv_k",
            "title": "Inverse mapping slope",
            "xlabel": "1/k (deg/dB)",
        },
        {
            "x": "JND_predicted_sigma_over_k",
            "title": "Full prediction",
            "xlabel": "Sigma/k (deg)",
        },
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11, 3.6),
        sharey=True,
    )

    y = table["JND_measured"]

    y_max = np.nanmax(y) * 1.1

    for ax, spec in zip(axes, plot_specs):

        x_col = spec["x"]

        for cue in CUE_ORDER:
            cue_df = table[table["reference_cue"] == cue]

            if cue_df.empty:
                continue

            ax.scatter(
                cue_df[x_col],
                cue_df["JND_measured"],
                color=CUE_COLORS.get(cue),
                label=cue,
                alpha=0.9,
            )

        fit = fit_simple_linear_prediction(
            x=table[x_col],
            y=table["JND_measured"],
        )

        x_vals = pd.to_numeric(table[x_col], errors="coerce")
        x_min = np.nanmin(x_vals)
        x_max = np.nanmax(x_vals)

        if np.isfinite(fit["slope"]):
            x_line = np.linspace(x_min, x_max, 100)
            y_line = fit["intercept"] + fit["slope"] * x_line

            ax.plot(
                x_line,
                y_line,
                color="black",
                linewidth=1.5,
                alpha=0.8,
            )

        ax.text(
            0.05,
            0.95,
            f"R²={fit['r2']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

        ax.set_title(spec["title"])
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylim(0, y_max)

    axes[0].set_ylabel("Measured JND (deg)")

    handles, labels = axes[-1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    if unique:
        fig.legend(
            unique.values(),
            unique.keys(),
            title="Cue",
            loc="center right",
            frameon=True,
        )
        fig.subplots_adjust(right=0.86)

    fig.suptitle(
        "Do JNDs follow uncertainty or mapping slope?",
        y=1.03,
        fontsize=15,
    )

    fig.tight_layout()

    if save:
        if save_path is None:
            save_dir = derivatives_root / "figures"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "jnd_component_triptych.png"

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved component triptych to: {save_path}")

    return fig, axes, table


def plot_component_model_comparison(
    subject_id=None,
    derivatives_root="data/psychometrics",
    cues=("ILD", "ITD", "COMBINED"),
    save=False,
    save_path=None,
):
    """
    Plot R² comparison for:
        sigma only
        1/k only
        sigma/k
    """

    derivatives_root = Path(derivatives_root)

    table_path = derivatives_root / "jnd_component_summary.csv"

    if table_path.exists():
        table = pd.read_csv(table_path)
    else:
        table = make_jnd_component_table(
            subject_id=subject_id,
            derivatives_root=derivatives_root,
            cues=cues,
            save=True,
        )

    if subject_id is not None:
        table = table[table["subject_id"].astype(str) == str(subject_id)]

    table["reference_cue"] = table["reference_cue"].astype(str).str.upper()
    table = table[table["reference_cue"].isin([cue.upper() for cue in cues])]

    model_comparison = compare_jnd_component_models(table)

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    ax.bar(
        model_comparison["label"],
        model_comparison["r2"],
        color="lightgray",
        edgecolor="black",
    )

    ax.set_ylabel("R²")
    ax.set_ylim(0, 1)
    ax.set_title("Predicting measured JND")

    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()

    if save:
        if save_path is None:
            save_dir = derivatives_root / "figures"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "jnd_component_model_comparison.png"

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved model comparison plot to: {save_path}")

    return fig, ax, model_comparison


def plot_jnd_vs_k_by_subject(
    derivatives_root="data/psychometrics",
    cues=("ILD", "ITD", "COMBINED"),
    ncols=4,
    save=False,
    save_path=None,
):
    """
    Plot measured JND against k slope separately for each subject.

    Each point represents one cue × frequency condition.

    For each subject, fit:

        JND = intercept + B / k

    across all cues and frequencies together.

    The subplot annotation reports:
        B
        p-value for the inverse-k term
        R²
    """

    derivatives_root = Path(derivatives_root)

    table_path = derivatives_root / "jnd_component_summary.csv"

    if table_path.exists():
        table = pd.read_csv(table_path)
    else:
        table = make_jnd_component_table(
            derivatives_root=derivatives_root,
            cues=cues,
            save=True,
        )

    # ------------------------------------------------------------
    # Clean data
    # ------------------------------------------------------------

    table = table.copy()

    table["reference_cue"] = (
        table["reference_cue"]
        .astype(str)
        .str.upper()
    )

    table = table[
        table["reference_cue"].isin(
            [cue.upper() for cue in cues]
        )
    ]

    table["k_slope"] = pd.to_numeric(
        table["k_slope"],
        errors="coerce",
    )

    table["JND_measured"] = pd.to_numeric(
        table["JND_measured"],
        errors="coerce",
    )

    table = table.replace([np.inf, -np.inf], np.nan)

    table = table.dropna(
        subset=[
            "subject_id",
            "reference_cue",
            "k_slope",
            "JND_measured",
        ]
    )

    # Inverse model only makes sense for positive k
    table = table[table["k_slope"] > 0]

    if table.empty:
        raise ValueError("No valid JND/k-slope data found.")

    # ------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------

    subjects = list(table["subject_id"].astype(str).unique())

    n_subjects = len(subjects)
    nrows = int(np.ceil(n_subjects / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.0 * ncols, 3.6 * nrows),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    axes_flat = axes.flatten()

    # Common limits across subjects
    global_k_max = table["k_slope"].max()
    global_jnd_max = table["JND_measured"].max()

    # ------------------------------------------------------------
    # Subject panels
    # ------------------------------------------------------------

    fit_rows = []

    for ax, subject in zip(axes_flat, subjects):

        subject_df = table[
            table["subject_id"].astype(str) == subject
        ].copy()

        # --------------------------
        # Scatter points
        # --------------------------

        for cue in CUE_ORDER:

            if cue not in [c.upper() for c in cues]:
                continue

            cue_df = subject_df[
                subject_df["reference_cue"] == cue
            ]

            if cue_df.empty:
                continue

            ax.scatter(
                cue_df["k_slope"],
                cue_df["JND_measured"],
                color=CUE_COLORS[cue],
                label=cue,
                s=35,
                alpha=0.9,
                zorder=3,
            )

        # --------------------------
        # Subject-specific inverse fit
        # --------------------------

        fit = fit_inverse_model(
            subject_df["k_slope"],
            subject_df["JND_measured"],
        )

        fit_rows.append(
            {
                "subject_id": subject,
                **fit,
            }
        )

        test = permutation_test_inverse(
            subject_df["k_slope"],
            subject_df["JND_measured"],
        )

        if np.isfinite(fit["B"]):

            k_min = subject_df["k_slope"].min()
            k_max = subject_df["k_slope"].max()

            # Only show fit across observed k range
            k_line = np.linspace(
                k_min,
                k_max,
                300,
            )

            jnd_line = inverse_model(
                k=k_line,
                a=fit["a"],
                B=fit["B"],
            )

            ax.plot(
                k_line,
                jnd_line,
                color="black",
                linewidth=1.5,
                zorder=2,
            )

        # --------------------------
        # Statistics
        # --------------------------

        if np.isfinite(test["p_value"]):

            if test["p_value"] < 0.001:
                p_text = "p < .001"
            else:
                p_text = f"p = {test['p_value']:.3f}"

            stats_text = (
                f"B = {fit['B']:.2f}\n"
                f"{p_text}\n"
                f"R² = {fit['r2']:.2f}"
            )

        else:
            stats_text = (
                "B = NA\n"
                "p = NA\n"
                "R² = NA"
            )

        ax.text(
            0.96,
            0.95,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )

        ax.set_title(f"subject_id = {subject}")

        # ax.set_xlim(
        #     0,
        #     global_k_max * 1.05,
        # )
        #
        # ax.set_ylim(
        #     0,
        #     global_jnd_max * 1.08,
        # )

    # ------------------------------------------------------------
    # Remove unused panels
    # ------------------------------------------------------------

    for ax in axes_flat[n_subjects:]:
        ax.set_visible(False)

    # ------------------------------------------------------------
    # Shared labels
    # ------------------------------------------------------------

    fig.supxlabel("k slope (dB/degree)")
    fig.supylabel("Measured JND (degrees)")

    # One shared cue legend
    handles = [
        plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=CUE_COLORS[cue],
            label=cue,
        )
        for cue in CUE_ORDER
        if cue in [c.upper() for c in cues]
    ]

    fig.legend(
        handles=handles,
        title="Cue",
        loc="lower right",
        bbox_to_anchor=(0.99, 0.99),
    )

    fig.tight_layout(
        rect=[0.02, 0.02, 0.95, 0.98]
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    if save:

        if save_path is None:
            save_dir = derivatives_root.parent.parent / "analysis" / "figures"
            save_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_path = (
                save_dir
                / "jnd_vs_k_by_subject.png"
            )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Saved JND vs k plot to: {save_path}"
        )

    fit_summary = pd.DataFrame(fit_rows)

    return fig, axes, table, fit_summary


def make_jnd_model_table(
    subject_id=None,
    derivatives_root="data/psychometrics",
    save=True,
):
    """
    Create a tidy table for statistical modelling of JND components.

    One row = subject × cue × frequency.
    """

    table = make_jnd_component_table(
        subject_id=subject_id,
        derivatives_root=derivatives_root,
        save=False,
    )

    model_table = table.copy()

    model_table = model_table.rename(
        columns={
            "reference_cue": "cue",
            "JND_predicted_sigma_over_k": "JND_predicted",
        }
    )

    keep_cols = [
        "subject_id",
        "cue",
        "frequency",
        "JND_measured",
        "k_slope",
        "inv_k",
        "sigma_y",
        "JND_predicted",
        "prediction_error",
        "prediction_ratio",
        "n_k_points",
    ]

    keep_cols = [col for col in keep_cols if col in model_table.columns]
    model_table = model_table[keep_cols].copy()

    # Numeric cleanup
    numeric_cols = [
        "frequency",
        "JND_measured",
        "k_slope",
        "inv_k",
        "sigma_y",
        "JND_predicted",
        "prediction_error",
        "prediction_ratio",
        "n_k_points",
    ]

    for col in numeric_cols:
        if col in model_table.columns:
            model_table[col] = pd.to_numeric(model_table[col], errors="coerce")

    model_table = model_table.replace([np.inf, -np.inf], np.nan)

    model_table = model_table.dropna(
        subset=[
            "JND_measured",
            "k_slope",
            "inv_k",
            # "sigma_y",
        ]
    )

    if save:
        out_path = Path(derivatives_root) / "jnd_component_model_table.csv"
        model_table.to_csv(out_path, index=False)
        print(f"Saved JND component model table to: {out_path}")

    return model_table