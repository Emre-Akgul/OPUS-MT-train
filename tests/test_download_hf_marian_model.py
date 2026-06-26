import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_hf_marian_model.py"
spec = importlib.util.spec_from_file_location("download_hf_marian_model", SCRIPT)
download_hf_marian_model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = download_hf_marian_model
spec.loader.exec_module(download_hf_marian_model)


FILES = [
    ".gitattributes",
    "README.md",
    "best-bleu.decoder.yml",
    "opus.src.spm32k-model",
    "opus.trg.spm32k-model",
    "synthetic_parallel_16lang_1_4m_gemini_3_1_flash_lite.spm32k-spm32k.transformer.model1.npz.best-bleu.npz",
    "synthetic_parallel_16lang_1_4m_gemini_3_1_flash_lite.spm32k-spm32k.transformer.model1.npz.yml",
    "synthetic_parallel_16lang_1_4m_gemini_3_1_flash_lite.spm32k-spm32k.vocab.yml",
    "synthetic_parallel_16lang_1_4m_gemini_3_1_flash_lite.transformer.mk",
]


def test_parse_hf_reference_accepts_model_url():
    ref = download_hf_marian_model.parse_hf_reference(
        "https://huggingface.co/TartarusXXX/synthetic-en2n-marian/tree/main"
    )

    assert ref.repo_id == "TartarusXXX/synthetic-en2n-marian"
    assert ref.revision == "main"


def test_choose_defaults_for_opus_mt_style_repo():
    assert download_hf_marian_model.choose_model(FILES, None).endswith(".best-bleu.npz")
    assert (
        download_hf_marian_model.choose_file(
            FILES,
            None,
            [r".*\.vocab\.yml", r"vocab\.yml"],
            "vocab .yml",
        )
        == "synthetic_parallel_16lang_1_4m_gemini_3_1_flash_lite.spm32k-spm32k.vocab.yml"
    )
    assert (
        download_hf_marian_model.choose_file(
            FILES,
            None,
            [r".*\.src\.spm[0-9]+k-model", r".*\.src\..*spm.*model"],
            "source SentencePiece model",
        )
        == "opus.src.spm32k-model"
    )
    assert (
        download_hf_marian_model.choose_file(
            FILES,
            None,
            [r".*\.trg\.spm[0-9]+k-model", r".*\.trg\..*spm.*model"],
            "target SentencePiece model",
        )
        == "opus.trg.spm32k-model"
    )
