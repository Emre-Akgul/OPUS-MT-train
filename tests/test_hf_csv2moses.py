import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "hf_csv2moses.py"
spec = importlib.util.spec_from_file_location("hf_csv2moses", SCRIPT)
hf_csv2moses = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf_csv2moses)


def base_args(tmp_path):
    return SimpleNamespace(
        delimiter=",",
        encoding="utf-8",
        include_same_lang=False,
        keep_empty=False,
        nway_mode="one-to-many",
        nway_source_lang="en",
        nway_shuffle_buffer=0,
        nway_shuffle_seed=1,
        nway_train_src=str(tmp_path / "train.src"),
        nway_train_trg=str(tmp_path / "train.trg"),
        nway_metadata=str(tmp_path / "train.meta.json"),
        output_dir=str(tmp_path / "pair"),
        overwrite=True,
        skip_same_lang=True,
        target_label_map=[],
    )


def write_csv(path):
    path.write_text(
        "en,tr,kmr,apc\n"
        "hello,merhaba,silav,marhaba\n"
        "bye,gule gule,,yalla\n"
        ",bos kaynak,vala,farigh\n",
        encoding="utf-8",
    )


def read_lines(path):
    return Path(path).read_text(encoding="utf-8").splitlines()


def test_nway_one_to_many_writes_labels_metadata_and_skips_empty(tmp_path):
    csv_path = tmp_path / "sample.csv"
    write_csv(csv_path)
    args = base_args(tmp_path)
    args.target_label_map = ["apc:ar"]

    hf_csv2moses.convert_nway_train(
        csv_path,
        args,
        src_langs=["en"],
        trg_langs=["en", "tr", "kmr", "apc"],
    )

    assert read_lines(args.nway_train_src) == [
        ">>tr<< hello",
        ">>kmr<< hello",
        ">>ar<< hello",
        ">>tr<< bye",
        ">>ar<< bye",
    ]
    assert read_lines(args.nway_train_trg) == [
        "merhaba",
        "silav",
        "marhaba",
        "gule gule",
        "yalla",
    ]

    metadata = json.loads(Path(args.nway_metadata).read_text(encoding="utf-8"))
    assert metadata["source_lang"] == "en"
    assert metadata["target_langs"] == ["tr", "kmr", "apc"]
    assert metadata["target_label_map"] == {"apc": "ar"}
    assert metadata["written_by_target"] == {"apc": 2, "kmr": 1, "tr": 2}
    assert metadata["skipped_by_target"] == {"apc": 0, "kmr": 1, "tr": 0}
    assert metadata["skipped_source_empty"] == 3


def test_nway_shuffle_is_deterministic_for_fixed_seed(tmp_path):
    csv_path = tmp_path / "sample.csv"
    write_csv(csv_path)
    args = base_args(tmp_path)
    args.nway_shuffle_buffer = 2
    args.nway_shuffle_seed = 7

    hf_csv2moses.convert_nway_train(csv_path, args, ["en"], ["tr", "kmr"])
    first_src = read_lines(args.nway_train_src)
    first_trg = read_lines(args.nway_train_trg)

    hf_csv2moses.convert_nway_train(csv_path, args, ["en"], ["tr", "kmr"])

    assert read_lines(args.nway_train_src) == first_src
    assert read_lines(args.nway_train_trg) == first_trg


def test_nway_all_to_all_writes_directed_pairs_in_one_stream(tmp_path):
    csv_path = tmp_path / "sample.csv"
    write_csv(csv_path)
    args = base_args(tmp_path)
    args.nway_mode = "all-to-all"
    args.nway_source_lang = None

    hf_csv2moses.convert_nway_train(
        csv_path,
        args,
        src_langs=["en", "tr", "kmr"],
        trg_langs=["en", "tr", "kmr"],
    )

    assert read_lines(args.nway_train_src) == [
        ">>tr<< hello",
        ">>kmr<< hello",
        ">>en<< merhaba",
        ">>kmr<< merhaba",
        ">>en<< silav",
        ">>tr<< silav",
        ">>tr<< bye",
        ">>en<< gule gule",
        ">>kmr<< bos kaynak",
        ">>tr<< vala",
    ]
    assert read_lines(args.nway_train_trg) == [
        "merhaba",
        "silav",
        "hello",
        "silav",
        "hello",
        "merhaba",
        "gule gule",
        "bye",
        "vala",
        "bos kaynak",
    ]

    metadata = json.loads(Path(args.nway_metadata).read_text(encoding="utf-8"))
    assert metadata["mode"] == "all-to-all"
    assert metadata["source_lang"] is None
    assert metadata["language_pairs"] == [
        "en-tr",
        "en-kmr",
        "tr-en",
        "tr-kmr",
        "kmr-en",
        "kmr-tr",
    ]
    assert metadata["written_by_pair"] == {
        "en-kmr": 1,
        "en-tr": 2,
        "kmr-en": 1,
        "kmr-tr": 2,
        "tr-en": 2,
        "tr-kmr": 2,
    }
    assert metadata["skipped_source_empty_by_lang"] == {"en": 2, "kmr": 2, "tr": 0}
    assert metadata["skipped_by_pair"] == {
        "en-kmr": 1,
        "en-tr": 0,
        "kmr-en": 1,
        "kmr-tr": 0,
        "tr-en": 1,
        "tr-kmr": 1,
    }


def test_pairwise_conversion_still_writes_clean_gzip_files(tmp_path):
    csv_path = tmp_path / "sample.csv"
    write_csv(csv_path)
    args = base_args(tmp_path)

    hf_csv2moses.convert_csv(
        csv_path,
        args,
        corpus="sample",
        src_langs=["en"],
        trg_langs=["tr"],
    )

    src_path = tmp_path / "pair" / "sample.en-tr.clean.en.gz"
    trg_path = tmp_path / "pair" / "sample.en-tr.clean.tr.gz"
    with gzip.open(src_path, "rt", encoding="utf-8") as src_in:
        assert src_in.read().splitlines() == ["hello", "bye"]
    with gzip.open(trg_path, "rt", encoding="utf-8") as trg_in:
        assert trg_in.read().splitlines() == ["merhaba", "gule gule"]
