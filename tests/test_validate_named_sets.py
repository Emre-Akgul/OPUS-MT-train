import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_named_sets.py"


def test_validate_named_sets_writes_per_segment_scores(tmp_path):
    references = tmp_path / "ref.txt"
    hypotheses = tmp_path / "hyp.txt"
    manifest = tmp_path / "sets.tsv"
    log = tmp_path / "scores.tsv"

    references.write_text("a b c\na b\nx y z\nx y\n", encoding="utf-8")
    hypotheses.write_text("a b c\na b\nx z y\nx y\n", encoding="utf-8")
    manifest.write_text(
        "name\tstart\tend\n"
        "simple\t0\t2\n"
        "long\t2\t4\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(manifest),
            str(references),
            str(log),
            "bleu",
            "micro",
            str(hypotheses),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert float(result.stdout.strip()) > 0

    rows = log.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "time\thypothesis\tmetric\tset\tscore\tlines"
    assert any("\t__aggregate_micro__\t" in row for row in rows)
    assert any("\tsimple\t100.000000\t2" in row for row in rows)
    assert any("\tlong\t" in row and row.endswith("\t2") for row in rows)
