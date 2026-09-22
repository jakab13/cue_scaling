from pathlib import Path

from analysis.fit_psychometrics import fit_run_files


RAW_ROOT = Path(
    "data/frequency_alignment/raw"
)

PSYCHOMETRICS_ROOT = Path(
    "data/frequency_alignment/psychometrics"
)


def fit_all_alignment_data(
    overwrite=False,
):
    """
    Fit all frequency-alignment runs only.
    """

    csv_files = sorted(
        RAW_ROOT.rglob("*.csv")
    )

    if not csv_files:
        print(
            f"No raw files found in {RAW_ROOT}"
        )
        return

    print(
        f"Found {len(csv_files)} alignment run files."
    )

    fit_run_files(
        csv_paths=csv_files,
        derivatives_root=PSYCHOMETRICS_ROOT,
        overwrite=overwrite,
        analysis_label="frequency_alignment",
    )


if __name__ == "__main__":
    fit_all_alignment_data()