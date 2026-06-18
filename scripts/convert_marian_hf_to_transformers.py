#!/usr/bin/env python3

import argparse
import importlib.util
import json
import os
import re
import shutil
import urllib.request
from pathlib import Path


HF_LANGUAGE_ALIASES = {
    "multi": "multilingual",
    "multilingual": "multilingual",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Download a Marian/OPUS-MT style model repo from Hugging Face, "
            "convert it to Transformers MarianMT format, and optionally upload it."
        )
    )
    parser.add_argument("--source-repo", required=True, help="HF Marian repo id, e.g. TartarusXXX/kmr-en-marian.")
    parser.add_argument("--dest-repo", required=True, help="HF Transformers repo id to create/update.")
    parser.add_argument("--model", help="Model .npz filename/path inside source repo. Defaults to *best-bleu*.npz.")
    parser.add_argument("--decoder", help="Decoder YAML filename/path inside source repo. Defaults to *decoder.yml.")
    parser.add_argument("--vocab", help="Vocab YAML filename/path inside source repo. Defaults to *vocab.yml.")
    parser.add_argument("--source-spm", help="Source SentencePiece filename/path inside source repo.")
    parser.add_argument("--target-spm", help="Target SentencePiece filename/path inside source repo.")
    parser.add_argument("--source-lang", required=True, help="Source language code for README/tokenizer config.")
    parser.add_argument(
        "--target-lang",
        help="Single target language code. Use --target-langs for multilingual target models.",
    )
    parser.add_argument(
        "--target-langs",
        nargs="+",
        help="Target language codes for multilingual target models.",
    )
    parser.add_argument("--model-name", help="README title. Defaults to dest repo name.")
    parser.add_argument("--dataset", help="Training dataset name/source for README metadata.")
    parser.add_argument("--validation-set", help="Validation set name to mention if not already in source README.")
    parser.add_argument("--license", default="", help="HF model card license id, if any.")
    parser.add_argument(
        "--decoder-start-token-id",
        type=int,
        help="Override decoder_start_token_id in config.json and generation_config.json after conversion.",
    )
    parser.add_argument("--private", action="store_true", help="Create/update destination repo as private.")
    parser.add_argument("--revision", default="main", help="Source repo revision. Default: main.")
    parser.add_argument("--dest-revision", default="main", help="Destination branch/revision. Default: main.")
    parser.add_argument("--commit-message", default="Upload Marian model (best-bleu, transformers)")
    parser.add_argument("--token-env", default="HF_TOKEN", help="Environment variable containing HF token.")
    parser.add_argument("--env-file", default=".env", help="Optional env file to read HF_TOKEN from.")
    parser.add_argument(
        "--work-dir",
        default="work/hf-transformers",
        help="Working directory for downloaded, staged, and converted files.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove existing work dirs for this conversion.")
    parser.add_argument("--dry-run", action="store_true", help="Convert locally but do not upload.")
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


def require_runtime_deps():
    missing = []
    for module in ["huggingface_hub", "transformers", "torch", "sentencepiece", "safetensors", "yaml"]:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        pip_names = ["pyyaml" if name == "yaml" else name for name in missing]
        raise SystemExit(
            "Missing Python dependencies: "
            + ", ".join(missing)
            + "\nInstall them with:\n"
            + f"  pip install {' '.join(pip_names)}"
        )


def hf_language(code):
    return HF_LANGUAGE_ALIASES.get(code, code)


def target_langs(args):
    langs = []
    if args.target_lang:
        langs.append(args.target_lang)
    if args.target_langs:
        langs.extend(args.target_langs)
    deduped = []
    for lang in langs:
        if lang not in deduped:
            deduped.append(lang)
    if not deduped:
        raise SystemExit("Pass --target-lang or --target-langs.")
    return deduped


def target_label(args):
    langs = target_langs(args)
    return langs[0] if len(langs) == 1 else "multilingual"


def pair_tag(args):
    langs = target_langs(args)
    return f"{args.source_lang}-{langs[0] if len(langs) == 1 else 'multilingual'}"


def safe_dir_name(repo_id):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", repo_id)


def choose_file(root, explicit, patterns, label):
    if explicit:
        path = root / explicit
        if not path.exists():
            raise SystemExit(f"{label} not found in source repo: {explicit}")
        return path

    matches = []
    for pattern in patterns:
        matches.extend(root.glob(pattern))
    matches = sorted({p for p in matches if p.is_file()})
    if not matches:
        raise SystemExit(f"Could not infer {label}; pass it explicitly.")
    if len(matches) > 1:
        listed = "\n".join(f"  {p.relative_to(root)}" for p in matches)
        raise SystemExit(f"Multiple candidates for {label}; pass it explicitly:\n{listed}")
    return matches[0]


def choose_model(root, explicit):
    if explicit:
        return choose_file(root, explicit, [], "model .npz")
    best = sorted(root.glob("*best-bleu*.npz"))
    if len(best) == 1:
        return best[0]
    if len(best) > 1:
        listed = "\n".join(f"  {p.relative_to(root)}" for p in best)
        raise SystemExit(f"Multiple best-bleu models; pass --model explicitly:\n{listed}")
    return choose_file(root, None, ["*.npz"], "model .npz")


def choose_spm(root, explicit, side):
    if explicit:
        return choose_file(root, explicit, [], f"{side} SentencePiece model")

    if side == "source":
        patterns = ["source.spm", "*src*.spm*model", "*source*.spm*model"]
    else:
        patterns = ["target.spm", "*trg*.spm*model", "*target*.spm*model"]
    return choose_file(root, None, patterns, f"{side} SentencePiece model")


def copy_to(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src.resolve(), dst)
    return dst


def rewrite_decoder_for_conversion(src_decoder, dst_decoder, model_name, vocab_name):
    lines = Path(src_decoder).read_text(encoding="utf-8").splitlines()
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
        if stripped == "vocabs:":
            out.append("vocabs:")
            out.append(f"  - {vocab_name}")
            out.append(f"  - {vocab_name}")
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
    dst_decoder.write_text("\n".join(out) + "\n", encoding="utf-8")


def normalize_for_converter(source_dir, convert_input_dir, args):
    model = choose_model(source_dir, args.model)
    decoder = choose_file(source_dir, args.decoder, ["*decoder.yml"], "decoder YAML")
    vocab = choose_file(source_dir, args.vocab, ["*vocab.yml"], "vocab YAML")
    source_spm = choose_spm(source_dir, args.source_spm, "source")
    target_spm = choose_spm(source_dir, args.target_spm, "target")

    if convert_input_dir.exists():
        shutil.rmtree(convert_input_dir)
    convert_input_dir.mkdir(parents=True)

    copy_to(model, convert_input_dir / "model.npz")
    copy_to(vocab, convert_input_dir / vocab.name)
    copy_to(source_spm, convert_input_dir / "source.spm")
    copy_to(target_spm, convert_input_dir / "target.spm")
    rewrite_decoder_for_conversion(decoder, convert_input_dir / "decoder.yml", "model.npz", vocab.name)

    readme = source_dir / "README.md"
    if readme.exists():
        copy_to(readme, convert_input_dir / "SOURCE_README.md")

    return {
        "model": model,
        "decoder": decoder,
        "vocab": vocab,
        "source_spm": source_spm,
        "target_spm": target_spm,
        "source_readme": readme if readme.exists() else None,
    }


def load_marian_converter(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        from transformers.models.marian.convert_marian_to_pytorch import convert

        return convert
    except ModuleNotFoundError:
        pass

    converter_path = cache_dir / "convert_marian_to_pytorch.py"
    if not converter_path.exists():
        import transformers

        candidates = []
        version = getattr(transformers, "__version__", "")
        if version:
            candidates.append(f"v{version}")
        candidates.extend(["main", "v4.57.1", "v4.56.2"])

        errors = []
        for ref in candidates:
            url = (
                "https://raw.githubusercontent.com/huggingface/transformers/"
                f"{ref}/src/transformers/models/marian/convert_marian_to_pytorch.py"
            )
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    converter_path.write_bytes(response.read())
                break
            except Exception as exc:
                errors.append(f"{ref}: {exc}")
        else:
            raise SystemExit(
                "Could not import or download the Transformers Marian converter.\n"
                + "\n".join(errors)
            )

    patch_converter_source(converter_path)

    spec = importlib.util.spec_from_file_location("hf_marian_converter", converter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.convert


def patch_converter_source(converter_path):
    text = converter_path.read_text(encoding="utf-8")
    replacements = {
        "yaml_cfg = yaml.load(cfg_str[:-1], Loader=yaml.BaseLoader)": (
            'yaml_cfg = yaml.load(cfg_str.rstrip("\\0"), Loader=yaml.BaseLoader)'
        ),
        'yaml_cfg = yaml.load(cfg_str.rstrip("\\\\x00"), Loader=yaml.BaseLoader)': (
            'yaml_cfg = yaml.load(cfg_str.rstrip("\\0"), Loader=yaml.BaseLoader)'
        ),
    }
    patched = text
    for old, new in replacements.items():
        patched = patched.replace(old, new)
    if patched != text:
        converter_path.write_text(patched, encoding="utf-8")


def convert_model(convert_input_dir, output_dir):
    convert = load_marian_converter(output_dir.parent)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    convert(convert_input_dir, output_dir)
    ensure_safetensors(output_dir)


def ensure_safetensors(output_dir):
    if (output_dir / "model.safetensors").exists():
        bin_path = output_dir / "pytorch_model.bin"
        if bin_path.exists():
            bin_path.unlink()
        return

    from transformers import MarianMTModel

    model = MarianMTModel.from_pretrained(output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)
    bin_path = output_dir / "pytorch_model.bin"
    if bin_path.exists():
        bin_path.unlink()


def normalize_tokenizer_config(output_dir, source_lang, targets):
    path = output_dir / "tokenizer_config.json"
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data["source_lang"] = source_lang
    if len(targets) == 1:
        data["target_lang"] = targets[0]
    else:
        data["target_lang"] = "multilingual"
        data["target_langs"] = targets
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_generation_config_if_missing(output_dir):
    path = output_dir / "generation_config.json"
    if path.exists():
        return
    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = {
        "bos_token_id": config.get("bos_token_id"),
        "decoder_start_token_id": config.get("decoder_start_token_id"),
        "eos_token_id": config.get("eos_token_id"),
        "forced_eos_token_id": config.get("forced_eos_token_id"),
        "max_length": config.get("max_length", 512),
        "num_beams": config.get("num_beams", 6),
        "pad_token_id": config.get("pad_token_id"),
        "transformers_version": config.get("transformers_version"),
    }
    data = {k: v for k, v in data.items() if v is not None}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def override_decoder_start_token(output_dir, token_id):
    if token_id is None:
        return
    for name in ["config.json", "generation_config.json"]:
        path = output_dir / name
        data = json.loads(path.read_text(encoding="utf-8"))
        data["decoder_start_token_id"] = token_id
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def frontmatter(args):
    targets = target_langs(args)
    lines = ["---"]
    if args.license:
        lines.append(f"license: {args.license}")
    lines.append("language:")
    for lang in [args.source_lang] + targets:
        lines.append(f"- {hf_language(lang)}")
    lines.extend(
        [
            "tags:",
            "- translation",
            "- transformers",
            "- marian",
            "- opus-mt",
            f"- {pair_tag(args)}",
        ]
    )
    if args.dataset:
        lines.extend(["datasets:", f"- {args.dataset}"])
    lines.append("---")
    return "\n".join(lines)


def strip_frontmatter(text):
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            return parts[2].lstrip()
    return text


def write_transformers_readme(output_dir, source_readme, args):
    title = args.model_name or args.dest_repo.split("/")[-1]
    targets = target_langs(args)
    target_text = ", ".join(f"`{lang}`" for lang in targets)
    body = ""
    if source_readme and source_readme.exists():
        body = strip_frontmatter(source_readme.read_text(encoding="utf-8", errors="replace"))

    validation_note = ""
    if args.validation_set and "Validation set:" not in body and "Validation set" not in body:
        validation_note = f"\n- Validation set: `{args.validation_set}`\n"

    if not body:
        body = f"# {title}\n\nThis is a Transformers MarianMT conversion of `{args.source_repo}`.\n"
    else:
        body = re.sub(r"^# .*$", f"# {title}", body, count=1, flags=re.MULTILINE)
        if not body.startswith("# "):
            body = f"# {title}\n\n" + body

    usage = f"""

## Transformers Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "{args.dest_repo}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# This model was trained as `{args.source_lang}` -> {target_text}.
# If the Marian training data uses target-language tags, include the desired
# target tag in the source sentence, following the training convention.
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
```
"""
    conversion = f"""

## Conversion

Converted from the Marian checkpoint in `{args.source_repo}` using the Transformers Marian converter.
- Source language: `{args.source_lang}`
- Target languages: {target_text}
{validation_note}
"""
    readme = frontmatter(args) + "\n\n" + body.rstrip() + conversion + usage
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_gitattributes(output_dir):
    content = "\n".join(
        [
            "*.safetensors filter=lfs diff=lfs merge=lfs -text",
            "*.spm filter=lfs diff=lfs merge=lfs -text",
            "",
        ]
    )
    (output_dir / ".gitattributes").write_text(content, encoding="utf-8")


def download_source_repo(args, source_dir):
    from huggingface_hub import snapshot_download

    token = os.environ.get(args.token_env)
    if source_dir.exists() and args.overwrite:
        shutil.rmtree(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        snapshot_download(
            repo_id=args.source_repo,
            repo_type="model",
            revision=args.revision,
            token=token,
            local_dir=source_dir,
            local_dir_use_symlinks=False,
        )
    )


def upload_output(args, output_dir):
    from huggingface_hub import HfApi

    token = os.environ.get(args.token_env)
    api = HfApi(token=token)
    api.create_repo(args.dest_repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=args.dest_repo,
        repo_type="model",
        folder_path=str(output_dir),
        revision=args.dest_revision,
        commit_message=args.commit_message,
    )


def main():
    args = parse_args()
    load_env_file(args.env_file)
    require_runtime_deps()
    targets = target_langs(args)

    work_root = Path(args.work_dir) / safe_dir_name(args.dest_repo)
    source_dir = work_root / "source"
    convert_input_dir = work_root / "converter-input"
    output_dir = work_root / "transformers"

    source_dir = download_source_repo(args, source_dir)
    chosen = normalize_for_converter(source_dir, convert_input_dir, args)
    convert_model(convert_input_dir, output_dir)
    normalize_tokenizer_config(output_dir, args.source_lang, targets)
    write_generation_config_if_missing(output_dir)
    override_decoder_start_token(output_dir, args.decoder_start_token_id)
    write_transformers_readme(output_dir, chosen["source_readme"], args)
    write_gitattributes(output_dir)

    print(f"Converted repo is staged at {output_dir}")
    if args.dry_run:
        print("Dry run requested; not uploading.")
        return

    upload_output(args, output_dir)
    print(f"Uploaded Transformers repo to https://huggingface.co/{args.dest_repo}")


if __name__ == "__main__":
    raise SystemExit(main())
