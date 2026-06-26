#!/usr/bin/env python3

import argparse
import csv
import gzip
import itertools
import json
import os
import random
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
    parser.add_argument(
        "--nway-train-src",
        help=(
            "Write a direct N-way training source file instead of pair files. "
            "Each CSV row is read once and projected from one source language "
            "to the selected target languages with target labels."
        ),
    )
    parser.add_argument(
        "--nway-train-trg",
        help="Write the direct N-way training target file used with --nway-train-src.",
    )
    parser.add_argument(
        "--nway-mode",
        default="one-to-many",
        choices=["one-to-many", "all-to-all"],
        help="Direct N-way projection mode. Default: one-to-many.",
    )
    parser.add_argument(
        "--nway-source-lang",
        help="Source language column for direct N-way training. Defaults to the first source language.",
    )
    parser.add_argument(
        "--nway-metadata",
        help="Write JSON metadata with per-target counts and skipped examples.",
    )
    parser.add_argument(
        "--target-label-map",
        nargs="*",
        default=[],
        help=(
            "Target label remaps for N-way source labels, e.g. apc:ar arz=ar. "
            "Targets not listed use their own language code."
        ),
    )
    parser.add_argument(
        "--skip-same-lang",
        action="store_true",
        help="Skip N-way examples where the source and target language are the same.",
    )
    parser.add_argument(
        "--nway-shuffle-buffer",
        type=int,
        default=0,
        help=(
            "Shuffle direct N-way training examples in chunks of this many examples. "
            "Default: 0 disables converter-side shuffling."
        ),
    )
    parser.add_argument(
        "--nway-shuffle-seed",
        type=int,
        default=1,
        help="Random seed for --nway-shuffle-buffer. Default: 1.",
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


def csv_rows(csv_path, args, langs):
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
        validate_columns(reader.fieldnames, langs)
        yield from reader


def parse_target_label_map(items):
    label_map = {}
    for item in items or []:
        if not item:
            continue
        if ":" in item:
            lang, label = item.split(":", 1)
        elif "=" in item:
            lang, label = item.split("=", 1)
        else:
            raise SystemExit(
                "Invalid --target-label-map item "
                f"{item!r}; use LANG:LABEL or LANG=LABEL."
            )
        lang = lang.strip()
        label = label.strip()
        if not lang or not label:
            raise SystemExit(
                "Invalid --target-label-map item "
                f"{item!r}; language and label must be non-empty."
            )
        label_map[lang] = label
    return label_map


def unique_in_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def nway_pairs(args, source_lang, src_langs, trg_langs):
    if args.nway_mode == "one-to-many":
        pairs = [(source_lang, trg) for trg in trg_langs]
    elif args.nway_mode == "all-to-all":
        pairs = [(src, trg) for src in src_langs for trg in trg_langs]
    else:
        raise SystemExit(f"Unsupported N-way mode: {args.nway_mode}")

    if not args.include_same_lang or args.skip_same_lang:
        pairs = [(src, trg) for src, trg in pairs if src != trg]
    if not pairs:
        raise SystemExit("No N-way language pairs selected.")
    return pairs


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
        rows = csv_rows(csv_path, args, sorted(set(src_langs + trg_langs)))
        try:
            for pair, (tmp_src, tmp_trg) in tmp_outputs.items():
                handles.append(
                    (
                        pair,
                        gzip.open(tmp_src, "wt", encoding="utf-8"),
                        gzip.open(tmp_trg, "wt", encoding="utf-8"),
                    )
                )

            for row in rows:
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
        finally:
            rows.close()
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


def convert_nway_train(csv_path, args, src_langs, trg_langs):
    if not args.nway_train_src or not args.nway_train_trg:
        raise SystemExit("Set both --nway-train-src and --nway-train-trg.")

    source_lang = args.nway_source_lang or src_langs[0]
    pairs = nway_pairs(args, source_lang, src_langs, trg_langs)
    pair_keys = [f"{src}-{trg}" for src, trg in pairs]
    required_langs = sorted({lang for pair in pairs for lang in pair})
    source_langs = unique_in_order(src for src, _ in pairs)
    target_langs = unique_in_order(trg for _, trg in pairs)
    label_map = parse_target_label_map(args.target_label_map)

    src_path = Path(args.nway_train_src)
    trg_path = Path(args.nway_train_trg)
    metadata_path = Path(args.nway_metadata) if args.nway_metadata else None
    output_paths = [src_path, trg_path]
    if metadata_path:
        output_paths.append(metadata_path)
    existing = [str(path) for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "N-way training output files already exist; use --overwrite to replace them: "
            + ", ".join(existing)
        )

    src_path.parent.mkdir(parents=True, exist_ok=True)
    trg_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_src = src_path.with_name(src_path.name + ".tmp")
    tmp_trg = trg_path.with_name(trg_path.name + ".tmp")
    tmp_metadata = (
        metadata_path.with_name(metadata_path.name + ".tmp") if metadata_path else None
    )
    written = {pair: 0 for pair in pair_keys}
    skipped = {pair: 0 for pair in pair_keys}
    written_by_source = {lang: 0 for lang in source_langs}
    written_by_target = {lang: 0 for lang in target_langs}
    skipped_by_target = {lang: 0 for lang in target_langs}
    skipped_source_empty_by_lang = {lang: 0 for lang in source_langs}
    skipped_source_empty = 0
    skipped_total = 0
    rng = random.Random(args.nway_shuffle_seed)
    buffer = []

    def write_buffer(src_out, trg_out):
        if not buffer:
            return
        rng.shuffle(buffer)
        for src_text, trg_text in buffer:
            src_out.write(src_text)
            trg_out.write(trg_text)
        buffer.clear()

    def write_example(src_out, trg_out, src_text, trg_text):
        if args.nway_shuffle_buffer > 0:
            buffer.append((src_text, trg_text))
            if len(buffer) >= args.nway_shuffle_buffer:
                write_buffer(src_out, trg_out)
        else:
            src_out.write(src_text)
            trg_out.write(trg_text)

    try:
        rows = csv_rows(csv_path, args, required_langs)
        try:
            with (
                tmp_src.open("w", encoding="utf-8") as src_out,
                tmp_trg.open("w", encoding="utf-8") as trg_out,
            ):
                row_count = 0
                for row in rows:
                    row_count += 1
                    row_texts = {
                        lang: clean_cell(row.get(lang, "")) for lang in required_langs
                    }
                    row_pairs = list(zip(pair_keys, pairs))
                    if args.nway_shuffle_buffer > 0:
                        rng.shuffle(row_pairs)
                    for pair_key, (src, trg) in row_pairs:
                        src_text = row_texts[src]
                        if not args.keep_empty and not src_text:
                            skipped_source_empty_by_lang[src] += 1
                            skipped_source_empty += 1
                            skipped_total += 1
                            continue
                        trg_text = row_texts[trg]
                        if not args.keep_empty and not trg_text:
                            skipped[pair_key] += 1
                            skipped_by_target[trg] += 1
                            skipped_total += 1
                            continue
                        label = label_map.get(trg, trg)
                        write_example(
                            src_out,
                            trg_out,
                            f">>{label}<< {src_text}\n",
                            trg_text + "\n",
                        )
                        written[pair_key] += 1
                        written_by_source[src] += 1
                        written_by_target[trg] += 1
                write_buffer(src_out, trg_out)
        finally:
            rows.close()
    except Exception:
        for path in (tmp_src, tmp_trg, tmp_metadata):
            if path is None:
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise

    os.replace(tmp_src, src_path)
    os.replace(tmp_trg, trg_path)
    metadata = {
        "mode": args.nway_mode,
        "source_lang": source_lang if args.nway_mode == "one-to-many" else None,
        "source_langs": source_langs,
        "target_langs": target_langs,
        "language_pairs": pair_keys,
        "target_label_map": label_map,
        "rows_read": row_count,
        "written_total": sum(written.values()),
        "written_by_pair": written,
        "written_by_source": written_by_source,
        "written_by_target": written_by_target,
        "skipped_total": skipped_total,
        "skipped_source_empty": skipped_source_empty,
        "skipped_source_empty_by_lang": skipped_source_empty_by_lang,
        "skipped_by_pair": skipped,
        "skipped_by_target": skipped_by_target,
        "shuffle_buffer": args.nway_shuffle_buffer,
        "shuffle_seed": args.nway_shuffle_seed,
        "source_path": str(src_path),
        "target_path": str(trg_path),
    }
    if metadata_path:
        tmp_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_metadata, metadata_path)

    total = sum(written.values())
    if args.nway_mode == "one-to-many":
        print(f"nway {source_lang}->targets: wrote {total} sentence pairs")
    else:
        print(f"nway all-to-all: wrote {total} sentence pairs")
    for pair_key in pair_keys:
        print(f"{pair_key}: wrote {written[pair_key]} sentence pairs")
    if skipped_total:
        if args.nway_mode == "one-to-many":
            print(f"nway {source_lang}->targets: skipped {skipped_total} empty examples")
        else:
            print(f"nway all-to-all: skipped {skipped_total} empty examples")
    print(src_path)
    print(trg_path)
    if metadata_path:
        print(metadata_path)


def main():
    args = parse_args()
    load_env_file(args.env_file)
    token = os.environ.get(args.token_env)
    headers = auth_headers(token)
    filename = detect_csv_file(args, headers)
    csv_path = download_hf_file(args, filename, headers)
    corpus = args.corpus or Path(filename).stem
    src_langs, trg_langs = resolve_langs(args)
    if args.nway_train_src or args.nway_train_trg:
        convert_nway_train(csv_path, args, src_langs, trg_langs)
    else:
        convert_csv(csv_path, args, corpus, src_langs, trg_langs)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
