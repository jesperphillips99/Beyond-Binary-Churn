#!/usr/bin/env python3
"""Convert the raw KKBox WSDM CSV files into the Parquet files the pipeline expects.

The modelling notebooks and ``truecut_precompute.py`` read the raw KKBox data from
``KKBoxData/RawData/*.parquet``. Kaggle distributes that data as (very large) CSV
files, so this script performs a *pure format conversion* – CSV to Parquet – with
no feature engineering, filtering, or scientific transformation of any kind. Every
column and value is preserved exactly; only the on-disk representation changes.

Expected inputs (place the Kaggle CSVs in ``KKBoxData/RawData/``):

    members_v3.csv
    transactions.csv
    transactions_v2.csv
    user_logs.csv          (~30 GB, ~392M rows – streamed in chunks)
    user_logs_v2.csv

Outputs (written next to the inputs):

    members_v3.parquet, transactions.parquet, transactions_v2.parquet,
    user_logs.parquet, user_logs_v2.parquet

Usage:
    python prepare_raw_data.py                 # convert every missing file
    python prepare_raw_data.py --force         # overwrite existing parquet
    python prepare_raw_data.py --only user_logs transactions
    python prepare_raw_data.py --chunksize 2000000

The conversion is memory-conscious: files are read in row chunks and appended to
the Parquet writer, so even ``user_logs`` converts within a few GB of RAM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RAW_DIR = Path(__file__).resolve().parent / "KKBoxData" / "RawData"

# Explicit output schemas so the Parquet files are identical regardless of the
# pandas type inference on any particular machine.
_TXN_SCHEMA = pa.schema([
    ("msno", pa.string()),
    ("payment_method_id", pa.int64()),
    ("payment_plan_days", pa.int64()),
    ("plan_list_price", pa.int64()),
    ("actual_amount_paid", pa.int64()),
    ("is_auto_renew", pa.int64()),
    ("transaction_date", pa.int64()),
    ("membership_expire_date", pa.int64()),
    ("is_cancel", pa.int64()),
])

_LOG_SCHEMA = pa.schema([
    ("msno", pa.string()),
    ("date", pa.int64()),
    ("num_25", pa.int64()),
    ("num_50", pa.int64()),
    ("num_75", pa.int64()),
    ("num_985", pa.int64()),
    ("num_100", pa.int64()),
    ("num_unq", pa.int64()),
    ("total_secs", pa.float64()),
])

_MEMBERS_SCHEMA = pa.schema([
    ("msno", pa.string()),
    ("city", pa.int64()),
    ("bd", pa.int64()),
    ("gender", pa.string()),
    ("registered_via", pa.int64()),
    ("registration_init_time", pa.int64()),
])

# name -> (pyarrow schema, pandas read_csv dtype map)
DATASETS: dict[str, tuple[pa.Schema, dict]] = {
    "members_v3": (
        _MEMBERS_SCHEMA,
        {"msno": "string", "city": "Int64", "bd": "Int64",
         "gender": "string", "registered_via": "Int64",
         "registration_init_time": "Int64"},
    ),
    "transactions": (
        _TXN_SCHEMA,
        {"msno": "string", "payment_method_id": "Int64", "payment_plan_days": "Int64",
         "plan_list_price": "Int64", "actual_amount_paid": "Int64",
         "is_auto_renew": "Int64", "transaction_date": "Int64",
         "membership_expire_date": "Int64", "is_cancel": "Int64"},
    ),
    "transactions_v2": (
        _TXN_SCHEMA,
        {"msno": "string", "payment_method_id": "Int64", "payment_plan_days": "Int64",
         "plan_list_price": "Int64", "actual_amount_paid": "Int64",
         "is_auto_renew": "Int64", "transaction_date": "Int64",
         "membership_expire_date": "Int64", "is_cancel": "Int64"},
    ),
    "user_logs": (
        _LOG_SCHEMA,
        {"msno": "string", "date": "Int64", "num_25": "Int64", "num_50": "Int64",
         "num_75": "Int64", "num_985": "Int64", "num_100": "Int64",
         "num_unq": "Int64", "total_secs": "float64"},
    ),
    "user_logs_v2": (
        _LOG_SCHEMA,
        {"msno": "string", "date": "Int64", "num_25": "Int64", "num_50": "Int64",
         "num_75": "Int64", "num_985": "Int64", "num_100": "Int64",
         "num_unq": "Int64", "total_secs": "float64"},
    ),
}


def _convert_one(name: str, chunksize: int, force: bool) -> None:
    schema, dtype = DATASETS[name]
    csv_path = RAW_DIR / f"{name}.csv"
    out_path = RAW_DIR / f"{name}.parquet"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing input CSV: {csv_path}\n"
            f"  Download the KKBox WSDM churn data from Kaggle and place "
            f"'{name}.csv' in {RAW_DIR}."
        )
    if out_path.exists() and not force:
        print(f"[skip]  {out_path.name} already exists (use --force to overwrite)")
        return

    print(f"[convert] {csv_path.name} -> {out_path.name}  (chunksize={chunksize:,})")
    writer: pq.ParquetWriter | None = None
    rows = 0
    tmp_path = out_path.with_suffix(".parquet.tmp")
    try:
        reader = pd.read_csv(csv_path, dtype=dtype, chunksize=chunksize)
        for i, chunk in enumerate(reader):
            # Preserve column order exactly as declared in the schema.
            chunk = chunk[[f.name for f in schema]]
            table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, schema, compression="snappy")
            writer.write_table(table)
            rows += len(chunk)
            print(f"    chunk {i:>3}  rows so far: {rows:,}", end="\r", flush=True)
    except Exception:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # Empty CSV (header only) – still emit a valid, empty Parquet file.
        pq.write_table(pa.Table.from_batches([], schema=schema), tmp_path,
                       compression="snappy")
    tmp_path.replace(out_path)
    print(f"\n[done]  {out_path.name}: {rows:,} rows")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(DATASETS),
                        help="Convert only the named dataset(s).")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing Parquet files.")
    parser.add_argument("--chunksize", type=int, default=5_000_000,
                        help="Rows per read chunk (default: 5,000,000).")
    args = parser.parse_args(argv)

    if not RAW_DIR.exists():
        print(f"error: raw data directory not found: {RAW_DIR}", file=sys.stderr)
        return 2

    targets = args.only or list(DATASETS)
    print(f"Raw data directory: {RAW_DIR}")
    print(f"Datasets to convert: {', '.join(targets)}\n")

    failures = 0
    for name in targets:
        try:
            _convert_one(name, args.chunksize, args.force)
        except FileNotFoundError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            failures += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[error] failed to convert {name}: {exc}", file=sys.stderr)
            failures += 1

    if failures:
        print(f"\nCompleted with {failures} failure(s). See messages above.",
              file=sys.stderr)
        return 1
    print("\nAll requested datasets converted successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
