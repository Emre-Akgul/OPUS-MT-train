#!/usr/bin/env python3

import argparse
import csv
import gzip
import itertools
import json
import os
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path


HF_BASE_URL = "https://huggingface.co"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download a CSV dataset from Hugging Face and convert language "
            "columns into OPUS-MT .clean.*.gz files."
        )
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face dataset repo id.")
    parser.add_argument(
        "--filename",
        help="CSV file in the dataset repo. If omitted, the single CSV file is detected.",
    )
    parser.add_argument("--revision", default="main", help="Dataset revision. Default: main.")
    parser.add_argument(
        "--repo-type",
        default="dataset",
        choices=["dataset", "model", "space"],
        help="Hugging Face repository type. Default: dataset.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        help="Language columns to use on both source and target sides.",
    )
    parser.add_argument("--src-langs", nargs="+", help="Source language columns.")
    parser.add_argument("--trg-langs", nargs="+", help="Target language columns.")
    parser.add_argument(
        "-c",
        "--corpus",
        help="Corpus name used in output filenames. Defaults to the CSV basename.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="work/data/simple",
        help="Output directory for .clean.*.gz files. Default: work/data/simple.",
    )
    parser.add_argument(
        "--cache-dir",
        default="work/data/hf",
        help="Directory for downloaded Hugging Face files. Default: work/data/hf.",
    )
    parser.add_argument("--delimiter", help="CSV delimiter. Defaults to auto-detection.")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding. Default: utf-8.")
    parser.add_argument("--env-file", default=".env", help="Env file to read HF_TOKEN from.")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hugging Face token. Default: HF_TOKEN.",
    )
    parser.add_argument(
        "--include-same-lang",
        action="store_true",
        help="Also create same-language pairs, using .clean.LANG1/.clean.LANG2 extensions.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep rows where either side is empty. Default: skip them.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs.")
    return parser.parse_args()


def load_env_file(path):
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_repo_path(repo_type, repo_id):
    if repo_type == "dataset":
        return f"datasets/{repo_id}"
    if repo_type == "space":
        return f"spaces/{repo_id}"
    return repo_id


def hf_api_repo_path(repo_type, repo_id):
    if repo_type == "dataset":
        return f"datasets/{repo_id}"
    if repo_type == "space":
        return f"spaces/{repo_id}"
    return f"models/{repo_id}"


def request_json(url, headers):
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def detect_csv_file(args, headers):
    if args.filename:
        return args.filename

    repo_path = hf_api_repo_path(args.repo_type, args.repo_id)
    revision = urllib.parse.quote(args.revision, safe="")
    url = f"{HF_BASE_URL}/api/{repo_path}/tree/{revision}?recursive=true"
    entries = request_json(url, headers)
    csv_files = [
        entry["path"]
        for entry in entries
        if entry.get("type") == "file"
        and entry.get("path", "").lower().endswith(".csv")
    ]
    csv_files = [
        path
        for path in csv_files
        if not path.lower().endswith(".report.csv")
        and ".report." not in path.lower()
    ]

    if not csv_files:
        raise SystemExit(f"No CSV files found in Hugging Face repo {args.repo_id!r}.")
    if len(csv_files) > 1:
        raise SystemExit(
            "Multiple CSV files found; set HF_CSV_FILE/--filename explicitly: "
            + ", ".join(csv_files)
        )
    return csv_files[0]


def safe_cache_name(repo_id, revision, filename):
    name = f"{repo_id}-{revision}-{filename}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def download_hf_file(args, filename, headers):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / safe_cache_name(args.repo_id, args.revision, filename)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    repo_path = hf_repo_path(args.repo_type, args.repo_id)
    quoted_revision = urllib.parse.quote(args.revision, safe="")
    quoted_filename = urllib.parse.quote(filename, safe="/")
    url = f"{HF_BASE_URL}/{repo_path}/resolve/{quoted_revision}/{quoted_filename}"
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")

    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    os.replace(tmp_path, cache_path)
    return cache_path


def clean_cell(value):
    return " ".join((value or "").replace("\r", " ").replace("\n", " ").split())


def detect_dialect(sample, delimiter):
    if delimiter:
        return csv.excel
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.excel


def sorted_langpair(src, trg):
    return "-".join(sorted([src, trg]))


def output_paths(output_dir, corpus, src, trg):
    if src == trg:
        return (
            output_dir / f"{corpus}.{src}-{trg}.clean.{src}1.gz",
            output_dir / f"{corpus}.{src}-{trg}.clean.{trg}2.gz",
        )
    langpair = sorted_langpair(src, trg)
    return (
        output_dir / f"{corpus}.{langpair}.clean.{src}.gz",
        output_dir / f"{corpus}.{langpair}.clean.{trg}.gz",
    )


def resolve_langs(args):
    src_langs = args.src_langs or args.langs
    trg_langs = args.trg_langs or args.langs
    if not src_langs or not trg_langs:
        raise SystemExit("Set --langs, or both --src-langs and --trg-langs.")
    return src_langs, trg_langs


def unique_pairs(src_langs, trg_langs, include_same):
    pairs = []
    seen = set()
    for src in src_langs:
        for trg in trg_langs:
            if src == trg and not include_same:
                continue
            key = (src, trg) if src == trg else tuple(sorted([src, trg]))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((src, trg))
    return pairs


def validate_columns(fieldnames, langs):
    missing = [lang for lang in langs if lang not in fieldnames]
    if missing:
        raise SystemExit(
            "CSV is missing language columns: "
            + ", ".join(missing)
            + ". Available columns: "
            + ", ".join(fieldnames)
        )


def convert_csv(csv_path, args, corpus, src_langs, trg_langs):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = unique_pairs(src_langs, trg_langs, args.include_same_lang)
    if not pairs:
        raise SystemExit("No language pairs selected.")

    outputs = {}
    existing = []
    for src, trg in pairs:
        src_path, trg_path = output_paths(output_dir, corpus, src, trg)
        outputs[(src, trg)] = (src_path, trg_path)
        if not args.overwrite:
            existing.extend(str(path) for path in (src_path, trg_path) if path.exists())
    if existing:
        expected = 2 * len(pairs)
        if len(existing) == expected:
            print("all expected output files already exist; use --overwrite to replace them")
            for pair in pairs:
                print(outputs[pair][0])
                print(outputs[pair][1])
            return
        raise SystemExit(
            "Some output files already exist; use --overwrite to replace them: "
            + ", ".join(existing)
        )

    tmp_outputs = {
        pair: (
            src_path.with_name(src_path.name + ".tmp"),
            trg_path.with_name(trg_path.name + ".tmp"),
        )
        for pair, (src_path, trg_path) in outputs.items()
    }
    handles = []
    written = {pair: 0 for pair in pairs}
    skipped = {pair: 0 for pair in pairs}

    try:
        with Path(csv_path).open("r", encoding=args.encoding, newline="") as input_file:
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
            validate_columns(reader.fieldnames, sorted(set(src_langs + trg_langs)))

            for pair, (tmp_src, tmp_trg) in tmp_outputs.items():
                handles.append(
                    (
                        pair,
                        gzip.open(tmp_src, "wt", encoding="utf-8"),
                        gzip.open(tmp_trg, "wt", encoding="utf-8"),
                    )
                )

            for row in reader:
                for pair, src_out, trg_out in handles:
                    src, trg = pair
                    src_text = clean_cell(row.get(src, ""))
                    trg_text = clean_cell(row.get(trg, ""))
                    if not args.keep_empty and (not src_text or not trg_text):
                        skipped[pair] += 1
                        continue
                    src_out.write(src_text + "\n")
                    trg_out.write(trg_text + "\n")
                    written[pair] += 1
    except Exception:
        for _, src_out, trg_out in handles:
            src_out.close()
            trg_out.close()
        for tmp_src, tmp_trg in tmp_outputs.values():
            for path in (tmp_src, tmp_trg):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        raise
    else:
        for _, src_out, trg_out in handles:
            src_out.close()
            trg_out.close()
        for pair, (src_path, trg_path) in outputs.items():
            tmp_src, tmp_trg = tmp_outputs[pair]
            os.replace(tmp_src, src_path)
            os.replace(tmp_trg, trg_path)

    for pair in pairs:
        src, trg = pair
        print(f"{src}-{trg}: wrote {written[pair]} sentence pairs")
        if skipped[pair]:
            print(f"{src}-{trg}: skipped {skipped[pair]} rows with an empty side")
        print(outputs[pair][0])
        print(outputs[pair][1])


def main():
    args = parse_args()
    load_env_file(args.env_file)
    token = os.environ.get(args.token_env)
    headers = auth_headers(token)
    filename = detect_csv_file(args, headers)
    csv_path = download_hf_file(args, filename, headers)
    corpus = args.corpus or Path(filename).stem
    src_langs, trg_langs = resolve_langs(args)
    convert_csv(csv_path, args, corpus, src_langs, trg_langs)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
