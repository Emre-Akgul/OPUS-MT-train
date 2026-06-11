#!/usr/bin/env python3

import argparse
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
BOUQUET_REPO = "facebook/bouquet"
BOUQUET_DEV_DIR = "data/sentence_level/dev"


DEFAULT_LANGS = [
    ("tur_Latn", "tr", "Türkçe"),
    ("eng_Latn", "en", "İngilizce"),
    ("deu_Latn", "de", "Almanca"),
    ("spa_Latn", "es", "İspanyolca"),
    ("fra_Latn", "fr", "Fransızca"),
    ("ell_Grek", "el", "Yunanca"),
    ("bul_Cyrl", "bg", "Bulgarca"),
    ("rus_Cyrl", "ru", "Rusça"),
    ("kat_Geor", "ka", "Gürcüce"),
    ("hye_Armn", "hy", "Ermenice"),
    ("pes_Arab", "fa", "Farsça"),
    ("ckb_Arab", "ckb", "Merkezi Kürtçe"),
    ("urd_Arab", "ur", "Urduca"),
    ("kmr_Latn", "kmr", "Kurmançça Kürtçe"),
]


TEXT_COLUMNS = ("src_text", "tgt_text", "text")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download BOUQuET sentence-level dev parquet files from Hugging Face "
            "and convert selected languages to OPUS-MT/Moses .clean.*.gz bitexts."
        )
    )
    parser.add_argument("--repo-id", default=BOUQUET_REPO, help="Hugging Face dataset repo id.")
    parser.add_argument("--revision", default="main", help="Dataset revision. Default: main.")
    parser.add_argument(
        "--split-dir",
        default=BOUQUET_DEV_DIR,
        help=f"Path inside repo containing per-language parquet files. Default: {BOUQUET_DEV_DIR}.",
    )
    parser.add_argument(
        "--langs",
        nargs="+",
        help=(
            "Languages to convert as BOUQuET_CODE[:OUTPUT_CODE]. "
            "Default is the Turkish-name list from the sample config."
        ),
    )
    parser.add_argument(
        "--src-langs",
        nargs="+",
        help="Output source language codes to include. Defaults to all selected languages.",
    )
    parser.add_argument(
        "--trg-langs",
        nargs="+",
        help="Output target language codes to include. Defaults to all selected languages.",
    )
    parser.add_argument(
        "--corpus-prefix",
        default="bouquet-dev",
        help="Prefix for output corpus names. Default: bouquet-dev.",
    )
    parser.add_argument(
        "--corpus-modes",
        nargs="+",
        default=["pair", "target"],
        choices=["pair", "target"],
        help=(
            "Output corpus naming modes. 'pair' writes bouquet-dev-en-tr style "
            "corpora; 'target' writes bouquet-dev-tr style corpora. Default: both."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="work/data/simple",
        help="Output directory for OPUS-MT .clean.*.gz files. Default: work/data/simple.",
    )
    parser.add_argument(
        "--cache-dir",
        default="work/data/hf/bouquet",
        help="Directory for downloaded parquet files. Default: work/data/hf/bouquet.",
    )
    parser.add_argument("--env-file", default=".env", help="Env file to read HF_TOKEN from.")
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable containing the Hugging Face token. Default: HF_TOKEN.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep rows where either side is empty. Default: skip them.",
    )
    parser.add_argument(
        "--write-map",
        default="",
        help="Optional path to write suggested TESTSET_BY_LANGPAIR/TRGLANG settings.",
    )
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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"} if token else {}


def hf_repo_path(repo_id):
    return f"datasets/{repo_id}"


def safe_cache_name(repo_id, revision, filename):
    name = f"{repo_id}-{revision}-{filename}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def download_hf_file(repo_id, revision, filename, cache_dir, headers):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / safe_cache_name(repo_id, revision, filename)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path

    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_filename = urllib.parse.quote(filename, safe="/")
    url = f"{HF_BASE_URL}/{hf_repo_path(repo_id)}/resolve/{quoted_revision}/{quoted_filename}"
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except Exception as exc:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        message = (
            f"Failed to download {filename} from {repo_id}. "
            "BOUQuET is gated; make sure HF_TOKEN is set and access was accepted. "
            f"Original error: {exc}"
        )
        if filename.endswith("/arb_Arab.parquet"):
            message = (
                f"Failed to download {filename} from {repo_id}. "
                "BOUQuET sentence-level dev does not publish Standard Arabic "
                "(arb_Arab.parquet) in this revision. Do not map apc_Arab or "
                "arz_Arab to ar unless you intentionally want dialectal Arabic. "
                f"Original error: {exc}"
            )
        raise SystemExit(message)

    os.replace(tmp_path, cache_path)
    return cache_path


def read_parquet_rows(path):
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise SystemExit(
            "Reading BOUQuET requires pyarrow. Install it in this environment "
            "with `python3 -m pip install pyarrow`, then rerun. "
            f"Original import error: {exc}"
        )

    table = pq.read_table(path)
    return table.to_pylist()


def clean_text(value):
    return " ".join((value or "").replace("\r", " ").replace("\n", " ").split())


def parse_lang_specs(specs):
    if not specs:
        return DEFAULT_LANGS

    parsed = []
    for spec in specs:
        if ":" in spec:
            bouquet_code, output_code = spec.split(":", 1)
        elif "=" in spec:
            bouquet_code, output_code = spec.split("=", 1)
        else:
            bouquet_code = spec
            output_code = bouquet_code.split("_", 1)[0]
        parsed.append((bouquet_code.strip(), output_code.strip(), bouquet_code.strip()))
    return parsed


def sorted_langpair(src, trg):
    return "-".join(sorted([src, trg]))


def output_paths(output_dir, corpus, src, trg):
    langpair = sorted_langpair(src, trg)
    return (
        output_dir / f"{corpus}.{langpair}.clean.{src}.gz",
        output_dir / f"{corpus}.{langpair}.clean.{trg}.gz",
    )


def row_text_for_lang(row, bouquet_code):
    if row.get("src_lang") == bouquet_code:
        return row.get("src_text")
    if row.get("tgt_lang") == bouquet_code:
        return row.get("tgt_text")
    for column in TEXT_COLUMNS:
        if column in row and row[column]:
            return row[column]
    return ""


def load_language(repo_id, revision, split_dir, bouquet_code, cache_dir, headers):
    filename = f"{split_dir.rstrip('/')}/{bouquet_code}.parquet"
    path = download_hf_file(repo_id, revision, filename, cache_dir, headers)
    rows = read_parquet_rows(path)
    data = {}
    for row in rows:
        uniq_id = row.get("uniq_id")
        if not uniq_id:
            continue
        text = clean_text(row_text_for_lang(row, bouquet_code))
        if text:
            data[str(uniq_id)] = text
    if not data:
        raise SystemExit(f"No usable rows found in {filename}.")
    return data


def write_pair(output_dir, corpus, src_code, trg_code, src_rows, trg_rows, keep_empty, overwrite):
    output_dir.mkdir(parents=True, exist_ok=True)
    src_path, trg_path = output_paths(output_dir, corpus, src_code, trg_code)
    if not overwrite and src_path.exists() and trg_path.exists():
        return 0, 0, (src_path, trg_path), True

    tmp_src = src_path.with_name(src_path.name + ".tmp")
    tmp_trg = trg_path.with_name(trg_path.name + ".tmp")
    written = 0
    skipped = 0
    common_ids = sorted(set(src_rows) & set(trg_rows))
    try:
        with gzip.open(tmp_src, "wt", encoding="utf-8") as src_out, gzip.open(
            tmp_trg, "wt", encoding="utf-8"
        ) as trg_out:
            for uniq_id in common_ids:
                src_text = src_rows.get(uniq_id, "")
                trg_text = trg_rows.get(uniq_id, "")
                if not keep_empty and (not src_text or not trg_text):
                    skipped += 1
                    continue
                src_out.write(src_text + "\n")
                trg_out.write(trg_text + "\n")
                written += 1
        os.replace(tmp_src, src_path)
        os.replace(tmp_trg, trg_path)
    except Exception:
        for tmp in (tmp_src, tmp_trg):
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        raise
    return written, skipped, (src_path, trg_path), False


def write_mapping_file(path, prefix, src_langs, trg_langs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pair_entries = [
        f"{src}-{trg}:{prefix}-{src}-{trg}"
        for src in src_langs
        for trg in trg_langs
        if src != trg
    ]
    trg_entries = [f"{trg}:{prefix}-{trg}" for trg in trg_langs]
    payload = {
        "TESTSET_NAME": prefix,
        "TESTSET_BY_LANGPAIR": " ".join(pair_entries),
        "TESTSET_BY_TRGLANG": " ".join(trg_entries),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    load_env_file(args.env_file)
    token = os.environ.get(args.token_env)
    headers = auth_headers(token)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)

    lang_specs = parse_lang_specs(args.langs)
    bouquet_to_output = {bouquet_code: output_code for bouquet_code, output_code, _ in lang_specs}
    output_to_bouquet = {output_code: bouquet_code for bouquet_code, output_code, _ in lang_specs}

    selected_src = args.src_langs or sorted(output_to_bouquet)
    selected_trg = args.trg_langs or sorted(output_to_bouquet)
    unknown = sorted((set(selected_src) | set(selected_trg)) - set(output_to_bouquet))
    if unknown:
        raise SystemExit(f"Unknown output language code(s): {', '.join(unknown)}")

    needed_output_langs = sorted(set(selected_src) | set(selected_trg))
    datasets = {}
    for output_code in needed_output_langs:
        bouquet_code = output_to_bouquet[output_code]
        datasets[output_code] = load_language(
            args.repo_id,
            args.revision,
            args.split_dir,
            bouquet_code,
            cache_dir,
            headers,
        )
        print(f"loaded {output_code} ({bouquet_code}): {len(datasets[output_code])} rows")

    summary = []
    for src, trg in itertools.product(selected_src, selected_trg):
        if src == trg:
            continue
        corpora = []
        if "pair" in args.corpus_modes:
            corpora.append(f"{args.corpus_prefix}-{src}-{trg}")
        if "target" in args.corpus_modes:
            corpora.append(f"{args.corpus_prefix}-{trg}")
        for corpus in corpora:
            written, skipped, paths, existed = write_pair(
                output_dir,
                corpus,
                src,
                trg,
                datasets[src],
                datasets[trg],
                args.keep_empty,
                args.overwrite,
            )
            summary.append((corpus, src, trg, written, skipped, existed, paths))

    if args.write_map:
        write_mapping_file(args.write_map, args.corpus_prefix, selected_src, selected_trg)

    for corpus, src, trg, written, skipped, existed, paths in summary:
        status = "exists" if existed else f"wrote {written}, skipped {skipped}"
        print(f"{status}: {corpus} {src}->{trg}")
        print(paths[0])
        print(paths[1])


if __name__ == "__main__":
    main()
