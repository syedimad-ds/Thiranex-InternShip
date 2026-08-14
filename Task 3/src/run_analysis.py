"""Command-line runner for the complete Task 3 EDA project."""

from pathlib import Path

from analysis import business_tables
from data_cleaning import run_preparation_pipeline
from visualization import generate_figures


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    _, prepared, audit, log = run_preparation_pipeline(root / "data/raw/task3_data.csv", root / "data/processed/task3_prepared.csv")
    tables = business_tables(prepared)
    figures = generate_figures(prepared, tables, root / "reports/figures")
    print(f"Raw audit: shape={audit['shape']}, duplicates={audit['duplicate_rows']}, missing={audit['missing_values']}")
    print(f"Preparation: {log}")
    print("\nCategory performance:\n", tables["category"].round(3))
    print("\nRegional performance:\n", tables["region"].round(3))
    print("\nFigures:\n" + "\n".join(str(path) for path in figures))


if __name__ == "__main__":
    main()
