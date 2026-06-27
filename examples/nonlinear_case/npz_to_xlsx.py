"""Convert MMD summary NPZ files to Excel workbooks.

This script leaves the source ``.npz`` files untouched. For each input file it
creates up to three ``.xlsx`` files next to the source file:

* ``<stem>_timeseries.xlsx``: 1D arrays aligned by ``ts``.
* ``<stem>_scalars.xlsx``: metadata, aggregate errors, and per-seed errors.
* ``<stem>_raw_arrays.xlsx``: any arrays that do not fit the time-series table.

Usage:
    python examples/nonlinear_case/npz_to_xlsx.py
    python examples/nonlinear_case/npz_to_xlsx.py path/to/file.npz
    python examples/nonlinear_case/npz_to_xlsx.py path/to/*.npz --out-dir path/to/xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUTS = (
    "outputs/mmd_summary_dt0.01.npz",
    "outputs/mmd_summary_OnsagerReg_dt0.01.npz",
)


def scalar_to_python(value: np.ndarray) -> Any:
    item = value.item()
    if isinstance(item, np.generic):
        return item.item()
    return item


def shape_text(value: np.ndarray) -> str:
    return "scalar" if value.ndim == 0 else "x".join(str(x) for x in value.shape)


def dataframe_from_array(key: str, value: np.ndarray) -> pd.DataFrame:
    if value.ndim == 1:
        return pd.DataFrame({key: value})
    if value.ndim == 2:
        columns = [f"{key}_{i}" for i in range(value.shape[1])]
        return pd.DataFrame(value, columns=columns)

    flat = value.reshape(value.shape[0], -1)
    columns = [f"{key}_{i}" for i in range(flat.shape[1])]
    return pd.DataFrame(flat, columns=columns)


def parse_error_scalar(key: str, value: Any) -> dict[str, Any]:
    body = key.removeprefix("err_")

    if "_seed" in body:
        method, rest = body.split("_seed", 1)
        seed, metric = rest.split("_", 1)
        return {
            "kind": "seed",
            "method": method,
            "seed": int(seed),
            "metric": metric,
            "stat": "value",
            "value": value,
        }

    for stat in ("mean", "std", "n"):
        suffix = f"_{stat}"
        if body.endswith(suffix):
            method_metric = body[: -len(suffix)]
            method, metric = method_metric.rsplit("_", 1)
            return {
                "kind": "aggregate",
                "method": method,
                "seed": None,
                "metric": metric,
                "stat": stat,
                "value": value,
            }

    return {
        "kind": "other",
        "method": None,
        "seed": None,
        "metric": None,
        "stat": None,
        "value": value,
    }


def split_npz_tables(npz_path: Path) -> dict[str, pd.DataFrame]:
    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    catalog_rows = [
        {
            "key": key,
            "shape": shape_text(value),
            "dtype": str(value.dtype),
            "ndim": value.ndim,
        }
        for key, value in arrays.items()
    ]

    scalars = {
        key: scalar_to_python(value)
        for key, value in arrays.items()
        if value.ndim == 0
    }
    vectors = {
        key: value
        for key, value in arrays.items()
        if value.ndim == 1
    }
    other_arrays = {
        key: value
        for key, value in arrays.items()
        if value.ndim > 1
    }

    ts = vectors.get("ts")
    timeseries_columns: dict[str, np.ndarray] = {}
    if ts is not None:
        timeseries_columns["ts"] = ts
        for key, value in vectors.items():
            if key != "ts" and len(value) == len(ts):
                timeseries_columns[key] = value
    else:
        lengths = {len(value) for value in vectors.values()}
        if len(lengths) == 1 and vectors:
            timeseries_columns["index"] = np.arange(next(iter(lengths)))
            timeseries_columns.update(vectors)

    used_timeseries_keys = set(timeseries_columns)
    timeseries_df = pd.DataFrame(timeseries_columns)

    metadata_rows = []
    error_rows = []
    for key, value in scalars.items():
        if key.startswith("err_"):
            error_rows.append({"key": key, **parse_error_scalar(key, value)})
        else:
            metadata_rows.append({"key": key, "value": value})

    metadata_df = pd.DataFrame(metadata_rows, columns=["key", "value"])
    raw_scalars_df = pd.DataFrame(
        [{"key": key, "value": value} for key, value in scalars.items()],
        columns=["key", "value"],
    )
    error_long_df = pd.DataFrame(
        error_rows,
        columns=["key", "kind", "method", "seed", "metric", "stat", "value"],
    )

    aggregate_df = pd.DataFrame()
    seed_df = pd.DataFrame()
    if not error_long_df.empty:
        aggregate_long = error_long_df[error_long_df["kind"] == "aggregate"]
        if not aggregate_long.empty:
            aggregate_df = (
                aggregate_long.pivot_table(
                    index=["method", "metric"],
                    columns="stat",
                    values="value",
                    aggfunc="first",
                )
                .reset_index()
                .rename_axis(None, axis=1)
            )
            preferred = ["method", "metric", "mean", "std", "n"]
            aggregate_df = aggregate_df[
                [col for col in preferred if col in aggregate_df.columns]
            ]

        seed_long = error_long_df[error_long_df["kind"] == "seed"]
        if not seed_long.empty:
            seed_df = (
                seed_long.pivot_table(
                    index=["method", "seed"],
                    columns="metric",
                    values="value",
                    aggfunc="first",
                )
                .reset_index()
                .rename_axis(None, axis=1)
                .sort_values(["method", "seed"])
            )

    leftover_array_frames = []
    for key, value in vectors.items():
        if key not in used_timeseries_keys:
            leftover_array_frames.append((key, dataframe_from_array(key, value)))
    for key, value in other_arrays.items():
        leftover_array_frames.append((key, dataframe_from_array(key, value)))

    return {
        "catalog": pd.DataFrame(catalog_rows, columns=["key", "shape", "dtype", "ndim"]),
        "timeseries": timeseries_df,
        "metadata": metadata_df,
        "raw_scalars": raw_scalars_df,
        "errors_long": error_long_df,
        "errors_aggregate": aggregate_df,
        "errors_by_seed": seed_df,
        "leftover_arrays": leftover_array_frames,
    }


def write_sheet(writer: pd.ExcelWriter, name: str, df: pd.DataFrame) -> None:
    safe_name = name[:31]
    df.to_excel(writer, sheet_name=safe_name, index=False)

    worksheet = writer.sheets[safe_name]
    for i, column in enumerate(df.columns, start=1):
        width = min(max(len(str(column)) + 2, 12), 42)
        worksheet.column_dimensions[worksheet.cell(row=1, column=i).column_letter].width = width
    worksheet.freeze_panes = "A2"


def write_excel_files(npz_path: Path, out_dir: Path | None = None) -> list[Path]:
    npz_path = npz_path.resolve()
    target_dir = out_dir.resolve() if out_dir else npz_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    tables = split_npz_tables(npz_path)
    stem = npz_path.stem
    written: list[Path] = []

    timeseries_path = target_dir / f"{stem}_timeseries.xlsx"
    if not tables["timeseries"].empty:
        with pd.ExcelWriter(timeseries_path, engine="openpyxl") as writer:
            write_sheet(writer, "timeseries", tables["timeseries"])
            write_sheet(writer, "catalog", tables["catalog"])
        written.append(timeseries_path)

    scalars_path = target_dir / f"{stem}_scalars.xlsx"
    with pd.ExcelWriter(scalars_path, engine="openpyxl") as writer:
        if not tables["metadata"].empty:
            write_sheet(writer, "metadata", tables["metadata"])
        if not tables["errors_aggregate"].empty:
            write_sheet(writer, "errors_aggregate", tables["errors_aggregate"])
        if not tables["errors_by_seed"].empty:
            write_sheet(writer, "errors_by_seed", tables["errors_by_seed"])
        if not tables["errors_long"].empty:
            write_sheet(writer, "errors_long", tables["errors_long"])
        write_sheet(writer, "raw_scalars", tables["raw_scalars"])
        write_sheet(writer, "catalog", tables["catalog"])
    written.append(scalars_path)

    if tables["leftover_arrays"]:
        arrays_path = target_dir / f"{stem}_raw_arrays.xlsx"
        with pd.ExcelWriter(arrays_path, engine="openpyxl") as writer:
            for key, df in tables["leftover_arrays"]:
                write_sheet(writer, key, df)
            write_sheet(writer, "catalog", tables["catalog"])
        written.append(arrays_path)

    return written


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Convert one or more MMD summary NPZ files to XLSX files."
    )
    parser.add_argument(
        "npz_files",
        nargs="*",
        type=Path,
        help="NPZ files to convert. Defaults to the two nonlinear-case MMD summaries.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for generated XLSX files. Defaults to each NPZ file's directory.",
    )
    args = parser.parse_args()

    if not args.npz_files:
        args.npz_files = [script_dir / rel for rel in DEFAULT_INPUTS]

    return args


def main() -> None:
    args = parse_args()
    for npz_file in args.npz_files:
        if not npz_file.exists():
            raise FileNotFoundError(npz_file)
        written = write_excel_files(npz_file, args.out_dir)
        print(f"{npz_file}:")
        for path in written:
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
