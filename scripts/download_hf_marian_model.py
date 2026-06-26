#!/usr/bin/env python3
"""Download and stage an OPUS-MT/Marian model repository from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


HF_BASE_URL = "https://huggingface.co"


@dataclass(frozen=True)
class HfReference:
    repo_id: str
    revision: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Marian/OPUS-MT style model repo from Hugging Face and "
            "stage stable filenames for OPUS-MT-train fine-tuning."
        )
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help=(
            "HF model repo id or URL, e.g. TartarusXXX/synthetic-en2n-marian "
            "or https://huggingface.co/TartarusXXX/synthetic-en2n-marian/tree/main."
        ),
    )
    parser.add_argument("--revision", help="HF revision/branch. Defaults to URL revision or main.")
    parser.add_argument("--output-dir", required=True, help="Directory for staged files.")
    parser.add_argument("--model", help="Model .npz filename inside the HF repo.")
    parser.add_argument("--vocab", help="Vocab .yml filename inside the HF repo.")
    parser.add_argument("--source-spm", help="Source SentencePiece model filename inside the HF repo.")
    parser.add_argument("--target-spm", help="Target SentencePiece model filename inside the HF repo.")
    parser.add_argument("--decoder", help="Decoder YAML filename inside the HF repo.")
    parser.add_argument("--config", help="Training config filename inside the HF repo.")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing HF token.")
    parser.add_argument("--env-file", default=".env", help="Optional env file to load before downloading.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite staged files.")
    return parser.parse_args()


def load_env_file(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_hf_reference(value: str) -> HfReference:
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urllib.parse.urlparse(value)
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise SystemExit(f"Could not parse Hugging Face model URL: {value}")
        repo_id = "/".join(parts[:2])
        revision = None
        if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve"}:
            revision = parts[3]
        return HfReference(repo_id=repo_id, revision=revision)
    return HfReference(repo_id=value)


def auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def request_json(url: str, headers: dict[str, str]):
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            raise SystemExit(
                f"Access denied for {url}. Set HF_TOKEN or pass --token-env for private repos."
            ) from exc
        raise


def list_repo_files(repo_id: str, revision: str, headers: dict[str, str]) -> list[str]:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    url = f"{HF_BASE_URL}/api/models/{quoted_repo}/tree/{quoted_revision}?recursive=true"
    entries = request_json(url, headers)
    files = [
        entry.get("path", "")
        for entry in entries
        if entry.get("type") == "file" and entry.get("path")
    ]
    if not files:
        raise SystemExit(f"No files found in HF model repo {repo_id!r} at revision {revision!r}.")
    return files


def choose_file(files: list[str], explicit: str | None, patterns: list[str], label: str) -> str:
    if explicit:
        if explicit not in files:
            raise SystemExit(f"{label} not found in source repo: {explicit}")
        return explicit

    matches: list[str] = []
    for pattern in patterns:
        regex = re.compile(pattern)
        matches.extend(path for path in files if regex.fullmatch(path))
    matches = sorted(set(matches))
    if not matches:
        raise SystemExit(f"Could not infer {label}; pass it explicitly.")
    if len(matches) > 1:
        listed = "\n".join(f"  {path}" for path in matches)
        raise SystemExit(f"Multiple candidates for {label}; pass it explicitly:\n{listed}")
    return matches[0]


def choose_model(files: list[str], explicit: str | None) -> str:
    if explicit:
        return choose_file(files, explicit, [], "model .npz")

    best_bleu = sorted(
        path
        for path in files
        if path.endswith(".npz") and "best-bleu" in Path(path).name
    )
    if len(best_bleu) == 1:
        return best_bleu[0]
    if len(best_bleu) > 1:
        listed = "\n".join(f"  {path}" for path in best_bleu)
        raise SystemExit(f"Multiple best-bleu models; pass --model explicitly:\n{listed}")

    candidates = sorted(
        path
        for path in files
        if path.endswith(".npz")
        and not Path(path).name.endswith(".optimizer.npz")
        and ".optimizer." not in Path(path).name
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit("Could not infer model .npz; pass it explicitly.")
    listed = "\n".join(f"  {path}" for path in candidates)
    raise SystemExit(f"Multiple model .npz candidates; pass --model explicitly:\n{listed}")


def optional_choose(files: list[str], explicit: str | None, patterns: list[str], label: str) -> str | None:
    if explicit:
        return choose_file(files, explicit, [], label)
    matches: list[str] = []
    for pattern in patterns:
        regex = re.compile(pattern)
        matches.extend(path for path in files if regex.fullmatch(path))
    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    return None


def download_file(
    repo_id: str,
    revision: str,
    filename: str,
    output_path: Path,
    headers: dict[str, str],
    overwrite: bool,
) -> None:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_filename = urllib.parse.quote(filename, safe="/")
    url = f"{HF_BASE_URL}/{quoted_repo}/resolve/{quoted_revision}/{quoted_filename}?download=true"
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request) as response, tmp_path.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except urllib.error.HTTPError as exc:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        if exc.code in {401, 403}:
            raise SystemExit(
                f"Access denied while downloading {filename}. Set HF_TOKEN for private repos."
            ) from exc
        raise
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    if tmp_path.stat().st_size == 0:
        tmp_path.unlink()
        raise SystemExit(f"Downloaded empty file for {filename}.")
    os.replace(tmp_path, output_path)


def main() -> int:
    args = parse_args()
    load_env_file(args.env_file)
    reference = parse_hf_reference(args.repo_id)
    revision = args.revision or reference.revision or "main"
    token = os.environ.get(args.token_env)
    headers = auth_headers(token)

    files = list_repo_files(reference.repo_id, revision, headers)
    model = choose_model(files, args.model)
    vocab = choose_file(files, args.vocab, [r".*\.vocab\.yml", r"vocab\.yml"], "vocab .yml")
    source_spm = choose_file(
        files,
        args.source_spm,
        [
            r".*\.src\.spm[0-9]+k-model",
            r".*\.src\..*spm.*model",
            r"source\.spm",
            r"source\.spm-model",
            r".*source.*spm.*model",
        ],
        "source SentencePiece model",
    )
    target_spm = choose_file(
        files,
        args.target_spm,
        [
            r".*\.trg\.spm[0-9]+k-model",
            r".*\.trg\..*spm.*model",
            r"target\.spm",
            r"target\.spm-model",
            r".*target.*spm.*model",
        ],
        "target SentencePiece model",
    )
    decoder = optional_choose(files, args.decoder, [r".*decoder\.ya?ml"], "decoder YAML")
    config = optional_choose(files, args.config, [r".*\.mk", r".*\.ya?ml"], "training config")

    output_dir = Path(args.output_dir)
    staged = {
        "model": ("model.npz", model),
        "vocab": ("vocab.yml", vocab),
        "source_spm": ("source.spm", source_spm),
        "target_spm": ("target.spm", target_spm),
    }
    if decoder:
        staged["decoder"] = ("decoder.yml", decoder)
    if config:
        staged["config"] = (Path(config).name, config)

    for _, (stage_name, remote_name) in staged.items():
        download_file(
            reference.repo_id,
            revision,
            remote_name,
            output_dir / stage_name,
            headers,
            args.overwrite,
        )

    manifest = {
        "repo_id": reference.repo_id,
        "revision": revision,
        "files": {
            key: {"staged": stage_name, "remote": remote_name}
            for key, (stage_name, remote_name) in staged.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Staged HF Marian model {reference.repo_id}@{revision} in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
