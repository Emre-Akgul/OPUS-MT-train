#!/usr/bin/env python3
"""Marian validation helper for named subsets inside one validation corpus.

Marian's `translation` validator appends the hypothesis file path after
`--valid-script-args` and expects exactly one score on stdout. This script
returns one aggregate score for checkpointing and appends per-set BLEU scores
to a sidecar TSV log.
"""

from __future__ import annotations

import collections
import datetime as _dt
import math
import os
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Segment:
    name: str
    start: int
    end: int


def usage() -> str:
    return (
        "usage: validate_named_sets.py MANIFEST REFERENCES LOG "
        "RETURN_METRIC AGGREGATE HYPOTHESIS\n"
        "\n"
        "MANIFEST is TSV with columns: name, start, end. Ranges are zero-based "
        "and end-exclusive."
    )


def read_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return [line.rstrip("\n") for line in handle]


def read_manifest(path: str, total_lines: int) -> list[Segment]:
    segments: list[Segment] = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if lineno == 1 and fields[:3] == ["name", "start", "end"]:
                continue
            if len(fields) < 3:
                raise ValueError(f"{path}:{lineno}: expected at least 3 TSV fields")
            name = fields[0]
            try:
                start = int(fields[1])
                end = int(fields[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: start/end must be integers") from exc
            if start < 0 or end < start or end > total_lines:
                raise ValueError(
                    f"{path}:{lineno}: invalid range {start}:{end} for {total_lines} lines"
                )
            segments.append(Segment(name=name, start=start, end=end))
    if not segments:
        raise ValueError(f"{path}: no validation segments found")
    return segments


def tokenize(line: str) -> list[str]:
    return line.split()


def ngrams(tokens: Sequence[str], order: int) -> collections.Counter[tuple[str, ...]]:
    return collections.Counter(
        tuple(tokens[index : index + order])
        for index in range(0, max(0, len(tokens) - order + 1))
    )


def internal_bleu(hypotheses: Sequence[str], references: Sequence[str], max_order: int = 4) -> float:
    """Compute corpus BLEU with simple exponential smoothing.

    This fallback is used when SacreBLEU is not installed in the training
    environment. It uses whitespace tokenization, matching Marian's segmented
    validation files.
    """

    matches = [0] * max_order
    totals = [0] * max_order
    hyp_len = 0
    ref_len = 0

    for hyp_line, ref_line in zip(hypotheses, references):
        hyp_tokens = tokenize(hyp_line)
        ref_tokens = tokenize(ref_line)
        hyp_len += len(hyp_tokens)
        ref_len += len(ref_tokens)

        for order in range(1, max_order + 1):
            hyp_ngrams = ngrams(hyp_tokens, order)
            totals[order - 1] += sum(hyp_ngrams.values())
            if hyp_ngrams:
                ref_ngrams = ngrams(ref_tokens, order)
                overlap = hyp_ngrams & ref_ngrams
                matches[order - 1] += sum(overlap.values())

    if hyp_len == 0:
        return 0.0

    precisions: list[float] = []
    smooth = 1.0
    for matched, total in zip(matches, totals):
        if total == 0:
            continue
        if matched == 0:
            smooth *= 2.0
            precisions.append(1.0 / (smooth * total))
        else:
            precisions.append(matched / total)

    if not precisions:
        return 0.0

    geo_mean = math.exp(sum(math.log(precision) for precision in precisions) / len(precisions))
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    return 100.0 * brevity_penalty * geo_mean


def sacrebleu_score(hypotheses: Sequence[str], references: Sequence[str]) -> float | None:
    try:
        import sacrebleu  # type: ignore
    except Exception:
        return None

    try:
        return float(sacrebleu.corpus_bleu(hypotheses, [references], tokenize="none").score)
    except TypeError:
        try:
            return float(sacrebleu.corpus_bleu(hypotheses, [references]).score)
        except Exception:
            return None
    except Exception:
        return None


def bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    score = sacrebleu_score(hypotheses, references)
    if score is not None:
        return score
    return internal_bleu(hypotheses, references)


def metric_score(metric: str, hypotheses: Sequence[str], references: Sequence[str]) -> float:
    if metric != "bleu":
        raise ValueError(f"unsupported metric: {metric}")
    return bleu(hypotheses, references)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_log(
    path: str,
    hypothesis_path: str,
    metric: str,
    aggregate_name: str,
    aggregate_score: float,
    aggregate_lines: int,
    segment_scores: Iterable[tuple[Segment, float]],
) -> None:
    ensure_parent(path)
    exists = os.path.exists(path)
    timestamp = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as handle:
        if not exists:
            handle.write("time\thypothesis\tmetric\tset\tscore\tlines\n")
        handle.write(
            f"{timestamp}\t{hypothesis_path}\t{metric}\t{aggregate_name}\t"
            f"{aggregate_score:.6f}\t{aggregate_lines}\n"
        )
        for segment, score in segment_scores:
            handle.write(
                f"{timestamp}\t{hypothesis_path}\t{metric}\t{segment.name}\t"
                f"{score:.6f}\t{segment.end - segment.start}\n"
            )


def main(argv: list[str]) -> int:
    if len(argv) != 7:
        print(usage(), file=sys.stderr)
        return 2

    _, manifest_path, reference_path, log_path, return_metric, aggregate, hypothesis_path = argv
    return_metric = return_metric.lower()
    aggregate = aggregate.lower()

    if return_metric != "bleu":
        print("validate_named_sets.py currently supports RETURN_METRIC=bleu", file=sys.stderr)
        return 2
    if aggregate not in {"micro", "macro"}:
        print("AGGREGATE must be micro or macro", file=sys.stderr)
        return 2

    references = read_lines(reference_path)
    hypotheses = read_lines(hypothesis_path)
    if len(references) != len(hypotheses):
        raise ValueError(
            f"reference/hypothesis line mismatch: {len(references)} != {len(hypotheses)}"
        )

    segments = read_manifest(manifest_path, len(references))
    segment_scores = [
        (
            segment,
            metric_score(
                return_metric,
                hypotheses[segment.start : segment.end],
                references[segment.start : segment.end],
            ),
        )
        for segment in segments
    ]

    if aggregate == "macro":
        aggregate_score = sum(score for _, score in segment_scores) / len(segment_scores)
        aggregate_name = "__aggregate_macro__"
    else:
        aggregate_score = metric_score(return_metric, hypotheses, references)
        aggregate_name = "__aggregate_micro__"

    append_log(
        log_path,
        hypothesis_path,
        return_metric,
        aggregate_name,
        aggregate_score,
        len(hypotheses),
        segment_scores,
    )
    print(f"{aggregate_score:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv))
    except Exception as exc:
        print(f"validate_named_sets.py: {exc}", file=sys.stderr)
        raise SystemExit(1)
