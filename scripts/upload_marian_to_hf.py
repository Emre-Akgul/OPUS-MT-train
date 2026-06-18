#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


LANG_NAMES = {
    "ar": "Arabic",
    "apc": "Levantine Arabic",
    "arz": "Egyptian Arabic",
    "bg": "Bulgarian",
    "ckb": "Central Kurdish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "hy": "Armenian",
    "ka": "Georgian",
    "kmr": "Northern Kurdish",
    "ku": "Kurdish",
    "ru": "Russian",
    "tr": "Turkish",
    "ur": "Urdu",
}


@dataclass
class ValidationResult:
    timestamp: str
    epoch: str
    updates: str
    metric: str
    score: str
    line: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage and upload a Marian/OPUS-MT model repository to Hugging Face Hub."
    )
    parser.add_argument("--repo-id", required=True, help="HF repo id, e.g. TartarusXXX/kmr-en-marian.")
    parser.add_argument("--model", required=True, help="Marian model .npz file, usually *.best-bleu.npz.")
    parser.add_argument("--decoder", help="Decoder YAML. Defaults to MODEL.decoder.yml if present.")
    parser.add_argument("--vocab", action="append", default=[], help="Vocab YAML/file. Can be repeated.")
    parser.add_argument("--spm", action="append", default=[], help="SentencePiece model. Can be repeated.")
    parser.add_argument("--config", action="append", default=[], help="Training config/mk/yml file. Can be repeated.")
    parser.add_argument("--train-log", help="Marian train log to include.")
    parser.add_argument("--valid-log", help="Marian validation log used for README metrics.")
    parser.add_argument("--extra-file", action="append", default=[], help="Extra file to include. Can be repeated.")
    parser.add_argument("--source-lang", required=True, help="Source language code, e.g. kmr.")
    parser.add_argument("--target-lang", required=True, help="Target language code, e.g. en.")
    parser.add_argument("--source-name", help="Human-readable source language name.")
    parser.add_argument("--target-name", help="Human-readable target language name.")
    parser.add_argument("--dataset", help="Training dataset name/source for the README.")
    parser.add_argument("--validation-set", help="Validation set name for the README.")
    parser.add_argument("--eval-langs", nargs="*", help="Optional evaluation target languages for multilingual runs.")
    parser.add_argument("--license", default="", help="HF model card license id, if any.")
    parser.add_argument("--model-name", help="README title. Defaults to repo name.")
    parser.add_argument("--private", action="store_true", help="Create/update the HF repo as private.")
    parser.add_argument("--revision", default="main", help="Hub branch/revision to upload to. Default: main.")
    parser.add_argument("--commit-message", default="Upload Marian model", help="HF commit message.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing the HF token.")
    parser.add_argument("--env-file", default=".env", help="Optional env file to load before uploading.")
    parser.add_argument(
        "--stage-dir",
        help="Directory for staged repo files. Defaults to work/hf-upload/<repo-name>.",
    )
    parser.add_argument("--overwrite-stage", action="store_true", help="Replace an existing stage dir.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare files but do not upload.")
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


def require_file(path, label):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"{label} does not exist: {p}")
    return p


def copy_file(src, dst_dir, dst_name=None):
    src = Path(src)
    dst = dst_dir / (dst_name or src.name)
    shutil.copy2(src.resolve(), dst)
    return dst


def rel_file_list(paths):
    return "\n".join(f"- `{p.name}`" for p in paths)


def lang_name(code, provided=None):
    return provided or LANG_NAMES.get(code, code)


def hf_language_code(code):
    if code in {"multi", "multilingual"}:
        return "multilingual"
    return code


def detect_decoder(model_path):
    candidate = Path(str(model_path) + ".decoder.yml")
    return candidate if candidate.exists() else None


def parse_best_validation(valid_log):
    if not valid_log:
        return None
    path = Path(valid_log)
    if not path.exists():
        return None

    best = None
    pattern = re.compile(
        r"^\[(?P<ts>[^\]]+)\]\s+\[valid\]\s+Ep\.\s+(?P<ep>\S+)\s+:\s+"
        r"Up\.\s+(?P<up>\S+)\s+:\s+(?P<metric>\S+)\s+:\s+"
        r"(?P<score>[0-9.eE+-]+)\s+:\s+new best"
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            best = ValidationResult(
                timestamp=match.group("ts"),
                epoch=match.group("ep"),
                updates=match.group("up"),
                metric=match.group("metric"),
                score=match.group("score"),
                line=line,
            )
    return best


def parse_all_validation_scores(valid_log):
    if not valid_log:
        return []
    path = Path(valid_log)
    if not path.exists():
        return []

    rows = []
    pattern = re.compile(
        r"^\[(?P<ts>[^\]]+)\]\s+\[valid\]\s+Ep\.\s+(?P<ep>\S+)\s+:\s+"
        r"Up\.\s+(?P<up>\S+)\s+:\s+(?P<metric>\S+)\s+:\s+"
        r"(?P<score>[0-9.eE+-]+)\s+:\s+(?P<status>.*)$"
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        rows.append(
            {
                "timestamp": match.group("ts"),
                "epoch": match.group("ep"),
                "updates": match.group("up"),
                "metric": match.group("metric"),
                "score": match.group("score"),
                "status": match.group("status"),
            }
        )
    return rows


def rewrite_decoder(decoder_path, dst_path, model_name, vocab_names):
    text = Path(decoder_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "models:":
            out.append("models:")
            out.append(f"  - {model_name}")
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                i += 1
            continue
        if stripped == "vocabs:" and vocab_names:
            out.append("vocabs:")
            for name in vocab_names:
                out.append(f"  - {name}")
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                i += 1
            continue
        if stripped.startswith("relative-paths:"):
            out.append("relative-paths: true")
            i += 1
            continue
        out.append(lines[i])
        i += 1
    dst_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_gitattributes(stage_dir):
    content = "\n".join(
        [
            "*.npz filter=lfs diff=lfs merge=lfs -text",
            "*spm*-model filter=lfs diff=lfs merge=lfs -text",
            "",
        ]
    )
    (stage_dir / ".gitattributes").write_text(content, encoding="utf-8")


def write_translate_helper(stage_dir):
    content = '''#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Translate text with this Marian model repository.")
    parser.add_argument("--decoder", default="best-bleu.decoder.yml", help="Decoder YAML path.")
    parser.add_argument("--marian-decoder", default="marian-decoder", help="Path to marian-decoder.")
    parser.add_argument("--input", help="Input text file. Defaults to stdin.")
    return parser.parse_args()


def main():
    args = parse_args()
    decoder = Path(args.decoder)
    cmd = [args.marian_decoder, "-c", str(decoder)]
    if args.input:
        with open(args.input, "rb") as inp:
            return subprocess.call(cmd, stdin=inp)
    return subprocess.call(cmd, stdin=sys.stdin.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path = stage_dir / "translate_with_marian.py"
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def yaml_front_matter(args, pair_tag):
    lines = ["---"]
    if args.license:
        lines.append(f"license: {args.license}")
    lines.extend(
        [
            "language:",
            f"- {hf_language_code(args.source_lang)}",
            f"- {hf_language_code(args.target_lang)}",
            "tags:",
            "- translation",
            "- marian",
            "- opus-mt",
            f"- {pair_tag}",
        ]
    )
    if args.dataset:
        lines.extend(["datasets:", f"- {args.dataset}"])
    lines.append("---")
    return "\n".join(lines)


def write_readme(args, stage_dir, staged_files, decoder_name, best_result):
    pair_tag = f"{args.source_lang}-{args.target_lang}"
    source_name = lang_name(args.source_lang, args.source_name)
    target_name = lang_name(args.target_lang, args.target_name)
    title = args.model_name or args.repo_id.split("/")[-1]
    all_scores = parse_all_validation_scores(args.valid_log)
    validation_set = args.validation_set or "not specified"
    dataset = args.dataset or "not specified"

    metric_section = "No validation log was provided."
    if best_result:
        metric_section = "\n".join(
            [
                f"- Best `{best_result.metric}`: `{best_result.score}`",
                f"- Epoch: `{best_result.epoch}`",
                f"- Update: `{best_result.updates}`",
                f"- Timestamp: `{best_result.timestamp}`",
                f"- Validation set: `{validation_set}`",
                "",
                "Raw best line:",
                "",
                "```text",
                best_result.line,
                "```",
            ]
        )

    score_table = ""
    if all_scores:
        tail = all_scores[-12:]
        rows = [
            "| Timestamp | Epoch | Update | Metric | Score | Status |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
        for row in tail:
            rows.append(
                f"| {row['timestamp']} | {row['epoch']} | {row['updates']} | "
                f"{row['metric']} | {row['score']} | {row['status']} |"
            )
        score_table = "\n\nRecent validation results:\n\n" + "\n".join(rows)

    eval_langs = ""
    if args.eval_langs:
        eval_langs = "\n\nEvaluation target languages: " + ", ".join(f"`{x}`" for x in args.eval_langs)

    files = rel_file_list(staged_files)
    readme = f"""{yaml_front_matter(args, pair_tag)}

# {title}

This repository contains a Marian NMT checkpoint packaged in OPUS-MT style.

## Model Details

- Source language: `{args.source_lang}` ({source_name})
- Target language: `{args.target_lang}` ({target_name})
- Task: translation
- Framework: Marian NMT
- Training data: `{dataset}`
- Validation set: `{validation_set}`{eval_langs}

## Validation

{metric_section}{score_table}

## Files

{files}

## Usage

Install Marian and run:

```bash
python translate_with_marian.py --decoder {decoder_name} < input.txt > output.txt
```

The decoder YAML uses paths relative to this repository.
"""
    (stage_dir / "README.md").write_text(readme, encoding="utf-8")


def run_hf_upload(args, stage_dir):
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with `pip install huggingface_hub`."
        ) from exc

    token = os.environ.get(args.token_env)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(stage_dir),
        revision=args.revision,
        commit_message=args.commit_message,
    )


def main():
    args = parse_args()
    load_env_file(args.env_file)

    model = require_file(args.model, "model")
    decoder = require_file(args.decoder, "decoder") if args.decoder else detect_decoder(model)
    if not decoder:
        raise SystemExit("Decoder YAML not found. Pass --decoder explicitly.")

    vocab_paths = [require_file(path, "vocab") for path in args.vocab]
    spm_paths = [require_file(path, "spm") for path in args.spm]
    config_paths = [require_file(path, "config") for path in args.config]
    extra_paths = [require_file(path, "extra-file") for path in args.extra_file]
    train_log = require_file(args.train_log, "train-log") if args.train_log else None
    valid_log = require_file(args.valid_log, "valid-log") if args.valid_log else None

    default_stage = Path("work/hf-upload") / args.repo_id.split("/")[-1]
    stage_dir = Path(args.stage_dir) if args.stage_dir else default_stage
    if stage_dir.exists() and args.overwrite_stage:
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    staged = []
    staged_model = copy_file(model, stage_dir)
    staged.append(staged_model)

    staged_vocabs = [copy_file(path, stage_dir) for path in vocab_paths]
    staged.extend(staged_vocabs)
    staged_spms = [copy_file(path, stage_dir) for path in spm_paths]
    staged.extend(staged_spms)
    staged_configs = [copy_file(path, stage_dir) for path in config_paths]
    staged.extend(staged_configs)
    if train_log:
        staged.append(copy_file(train_log, stage_dir))
    if valid_log:
        staged.append(copy_file(valid_log, stage_dir))
    for path in extra_paths:
        staged.append(copy_file(path, stage_dir))

    decoder_name = "best-bleu.decoder.yml"
    rewrite_decoder(decoder, stage_dir / decoder_name, staged_model.name, [p.name for p in staged_vocabs])
    staged.append(stage_dir / decoder_name)
    staged.append(write_translate_helper(stage_dir))
    write_gitattributes(stage_dir)

    best_result = parse_best_validation(valid_log)
    write_readme(args, stage_dir, staged, decoder_name, best_result)

    print(f"Staged {len(staged) + 2} files in {stage_dir}")
    if args.dry_run:
        print("Dry run requested; not uploading.")
        return

    run_hf_upload(args, stage_dir)
    print(f"Uploaded {stage_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    raise SystemExit(main())
