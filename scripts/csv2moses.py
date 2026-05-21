#!/usr/bin/env python3

import argparse
import csv
import gzip
import itertools
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a two-column parallel CSV into OPUS-MT training files "
            "under work/data/simple."
        )
    )
    parser.add_argument("csv_path", help="Input CSV file.")
    parser.add_argument("-s", "--src", required=True, help="Source language code.")
    parser.add_argument("-t", "--trg", required=True, help="Target language code.")
    parser.add_argument(
        "-c",
        "--corpus",
        help="Corpus name used in output filenames. Defaults to the CSV basename.",
    )
    parser.add_argument(
        "--src-column",
        help=(
            "Source column name. Defaults to the source language code, then "
            "falls back to SRCLANGS/source/src."
        ),
    )
    parser.add_argument(
        "--trg-column",
        help=(
            "Target column name. Defaults to the target language code, then "
            "falls back to TRGLANGS/target/trg."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="work/data/simple",
        help="Output directory for .clean.*.gz files. Default: work/data/simple.",
    )
    parser.add_argument(
        "--delimiter",
        help="CSV delimiter. If omitted, the delimiter is detected from the file.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding. Default: utf-8.",
    )
    parser.add_argument(
        "--preserve-langpair-order",
        action="store_true",
        help=(
            "Use SRC-TRG in output filenames instead of OPUS-MT's sorted "
            "language-pair name."
        ),
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep rows where either side is empty. Default: skip them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def clean_cell(value):
    return " ".join((value or "").replace("\r", " ").replace("\n", " ").split())


def detect_dialect(sample, delimiter):
    if delimiter:
        return csv.excel
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.excel


def pick_column(fieldnames, requested, candidates, kind):
    if requested:
        if requested not in fieldnames:
            raise SystemExit(
                f"CSV is missing requested {kind} column {requested!r}. "
                f"Available columns: {', '.join(fieldnames)}"
            )
        return requested

    for candidate in candidates:
        if candidate in fieldnames:
            return candidate

    raise SystemExit(
        f"Could not find a {kind} column. Tried: {', '.join(candidates)}. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def output_paths(output_dir, corpus, src, trg, preserve_order):
    langpair = f"{src}-{trg}" if preserve_order else "-".join(sorted([src, trg]))
    return (
        output_dir / f"{corpus}.{langpair}.clean.{src}.gz",
        output_dir / f"{corpus}.{langpair}.clean.{trg}.gz",
    )


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"Input CSV does not exist: {csv_path}")

    corpus = args.corpus or csv_path.stem
    output_dir = Path(args.output_dir)
    src_path, trg_path = output_paths(
        output_dir, corpus, args.src, args.trg, args.preserve_langpair_order
    )

    if not args.overwrite:
        existing = [str(path) for path in (src_path, trg_path) if path.exists()]
        if existing:
            raise SystemExit(
                "Output file already exists; use --overwrite to replace it: "
                + ", ".join(existing)
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_src = src_path.with_name(src_path.name + ".tmp")
    tmp_trg = trg_path.with_name(trg_path.name + ".tmp")

    written = 0
    skipped = 0
    try:
        with csv_path.open("r", encoding=args.encoding, newline="") as input_file:
            sample = input_file.read(65536)
            dialect = detect_dialect(sample, args.delimiter)
            try:
                input_file.seek(0)
                csv_input = input_file
            except OSError:
                csv_input = itertools.chain(sample.splitlines(True), input_file)

            reader_args = {"dialect": dialect}
            if args.delimiter:
                reader_args["delimiter"] = args.delimiter
            reader = csv.DictReader(csv_input, **reader_args)
            if not reader.fieldnames:
                raise SystemExit(f"CSV has no header row: {csv_path}")

            src_column = pick_column(
                reader.fieldnames,
                args.src_column,
                [args.src, "SRCLANGS", "source", "src"],
                "source",
            )
            trg_column = pick_column(
                reader.fieldnames,
                args.trg_column,
                [args.trg, "TRGLANGS", "target", "trg"],
                "target",
            )

            with gzip.open(tmp_src, "wt", encoding="utf-8") as src_out, gzip.open(
                tmp_trg, "wt", encoding="utf-8"
            ) as trg_out:
                for row_number, row in enumerate(reader, start=2):
                    src_text = clean_cell(row.get(src_column, ""))
                    trg_text = clean_cell(row.get(trg_column, ""))
                    if not args.keep_empty and (not src_text or not trg_text):
                        skipped += 1
                        continue
                    src_out.write(src_text + "\n")
                    trg_out.write(trg_text + "\n")
                    written += 1
    except Exception:
        for path in (tmp_src, tmp_trg):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    os.replace(tmp_src, src_path)
    os.replace(tmp_trg, trg_path)

    print(f"wrote {written} sentence pairs")
    if skipped:
        print(f"skipped {skipped} rows with an empty source or target")
    print(src_path)
    print(trg_path)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
