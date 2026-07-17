"""Build the frozen LoRA eval set from the paper split + remote-rescore cascade.

Reads two kinds of input:

1. ``eval_paper_split.json`` -- the FROZEN paper-level holdout (108 eval
   papers out of 723, seed 20260714). Immutable: this tool verifies its
   sha256[:16] against the pinned value and hard-fails on any mismatch.
   It is never regenerated or reformatted here.
2. One or more ``pass_at_k.jsonl`` files from the remote-rescore cascade
   (``out/remote_rescore/tier{1,2,3,4}*/pass_at_k.jsonl``) -- passed in
   explicitly via ``--tier-outputs`` rather than discovered by a fixed
   glob, because the cascade may be partially landed at any given time
   (see docs/eval_harness_design.md). Re-running this tool as more tiers
   complete is expected and safe.

Emits, per docs/eval_harness_design.md's "Eval set composition" table:

  * ``eval_set.jsonl`` -- three slices, each record tagged with an
    ``eval_slice`` field:
      - ``eval_band``      eval-paper records with label == "band"
                            (the improvement metric lives here)
      - ``anchor_solved``  eval-paper records at *exactly* k/k correct
                            ("8/8") -- must STAY solved after LoRA
                            training (catastrophic-forgetting detector)
      - ``anchor_fail``    eval-paper records labelled "collapse" at
                            *exactly* 0/k correct ("0/8") -- must STAY
                            failed (memorization/contamination detector)
  * ``train_uids.txt`` -- uids of every non-eval-paper record labelled
    "band" across the given tier outputs (the final cascade band minus
    eval papers). The LoRA pipeline consumes ONLY this file.

Hard-fail conditions (raise :class:`EvalSetError`, never silently skip):

  * the split file's sha256[:16] does not match the pinned value;
  * (defense-in-depth) any train uid's arxiv_id resolves to an eval
    paper -- structurally prevented by the bucketing logic below, but
    re-asserted explicitly so a future refactor cannot reintroduce
    leakage silently;
  * any assembled eval-set record is missing a non-blank ``statement``
    or ``answer``.

Anchor slices are deliberately stricter than the upstream stage's own
label boundaries: ``label == "solved"`` alone covers any pass_at_k above
0.75 (e.g. 7/8 = 0.875), and ``label == "collapse"`` covers the whole
sub-band region with no dominant wrong attractor (not just n_correct ==
0). The design doc names the anchors "8/8" and "0/8" explicitly, so this
module additionally requires the exact rollout count -- see
``_is_perfect_solve`` / ``_is_total_collapse``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

# Pinned sha256[:16] of evalharness/data/eval_paper_split.json's exact
# bytes (seed 20260714, 108 eval papers / 723 universe). This is the one
# true production value -- NOT a CLI flag, so there is no footgun for
# accidentally pointing production at an unpinned split. Tests that need
# to exercise a synthetic split call build()/load_split() directly with
# their own expected_split_sha16.
EXPECTED_SPLIT_SHA256_16 = "110a4bf27320f2b1"

DEFAULT_SPLIT_PATH = Path("evalharness/data/eval_paper_split.json")
DEFAULT_OUTPUT_DIR = Path("evalharness/data")

EVAL_SLICE_BAND = "eval_band"
EVAL_SLICE_ANCHOR_SOLVED = "anchor_solved"
EVAL_SLICE_ANCHOR_FAIL = "anchor_fail"
EVAL_SLICE_VALUES = (EVAL_SLICE_BAND, EVAL_SLICE_ANCHOR_SOLVED, EVAL_SLICE_ANCHOR_FAIL)

# Labels as stamped by icepick's pass_at_k stage
# (icepick.processing.pass_at_k.base.LABEL_VALUES). Re-declared as plain
# strings rather than imported -- this sub-repo has zero import
# dependency on icepick (mirrors the src/posers/* pattern); parity is a
# documentation matter, not a code one, since we only ever read these as
# data off disk.
LABEL_BAND = "band"
LABEL_SOLVED = "solved"
LABEL_COLLAPSE = "collapse"

UNDERPOWERED_HINT_N = 25  # docs/eval_harness_design.md "Open items"


class EvalSetError(ValueError):
    """Raised when the eval set cannot be built safely.

    Every raise site in this module is one of the three HARD-FAIL
    conditions from docs/eval_harness_design.md: split sha mismatch,
    train/eval leakage, or an eval record missing statement/answer.
    There is no soft/warn mode for these -- a silent miss here poisons
    the number this harness exists to produce.
    """


@dataclass
class BuildResult:
    """Everything a caller (CLI or test) might want after a successful build."""

    eval_set_path: Path
    train_uids_path: Path
    counts: dict
    warnings: list = field(default_factory=list)


def sha256_16(path: Path) -> str:
    """First 16 hex chars of the file's sha256 -- the pin format this design uses."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_split(path: Path, expected_sha16: str = EXPECTED_SPLIT_SHA256_16) -> dict:
    """Load + integrity-check the frozen paper split.

    Hard-fails (``EvalSetError``) on any sha mismatch, and on a missing
    ``eval_papers`` list. ``FileNotFoundError`` if the path itself does
    not exist -- callers that want a single exception type can catch
    ``(EvalSetError, FileNotFoundError)``.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"eval paper split not found: {path} -- this file is a frozen, "
            "pre-committed artifact; it is never generated by this tool"
        )
    actual = sha256_16(path)
    if actual != expected_sha16:
        raise EvalSetError(
            f"eval_paper_split integrity check FAILED for {path}: "
            f"sha256[:16]={actual!r}, expected {expected_sha16!r}. This "
            "file is frozen (docs/eval_harness_design.md) -- it must never "
            "be regenerated, reformatted, or hand-edited. Restore the "
            "original bytes before rerunning."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("eval_papers"), list):
        raise EvalSetError(f"{path} is missing an 'eval_papers' list -- malformed split file")
    return data


def _iter_jsonl(path: Path) -> Iterator[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"tier-output file not found: {path} -- the remote-rescore "
            "cascade has not produced this file yet. Pass only "
            "--tier-outputs paths that currently exist on disk; rerun this "
            "tool once more tiers land (see out/remote_rescore/*/pass_at_k.jsonl)"
        )
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvalSetError(f"{path}:{lineno}: invalid JSON ({exc})") from None
            if not isinstance(row, dict):
                raise EvalSetError(f"{path}:{lineno}: expected a JSON object, got {type(row).__name__}")
            yield row


def _is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _n_total(record: dict) -> int:
    return (
        int(record.get("n_correct") or 0)
        + int(record.get("n_wrong") or 0)
        + int(record.get("n_degenerate") or 0)
    )


def _is_perfect_solve(record: dict) -> bool:
    """All rollouts correct -- the strict "8/8" anchor-solved criterion.

    Deliberately stricter than ``label == "solved"`` alone, which
    triggers for anything above pass_at_k > 0.75 (e.g. 7/8 = 0.875).
    """
    n_correct = record.get("n_correct")
    if n_correct is None:
        return False
    n_total = _n_total(record)
    return n_total > 0 and n_correct == n_total


def _is_total_collapse(record: dict) -> bool:
    """Zero rollouts correct -- the strict "0/8" anchor-fail criterion.

    Deliberately stricter than ``label == "collapse"`` alone: a record
    with n_correct == 0 but a dominant wrong attractor is labelled
    "misdirection" upstream, not "collapse", and is excluded here too
    (misdirection is a distinct failure mode the design doc scopes out
    of the anchor-fail slice).
    """
    n_correct = record.get("n_correct")
    if n_correct is None:
        return False
    return _n_total(record) > 0 and n_correct == 0


def _tag(record: dict, slice_name: str) -> dict:
    """Return a NEW dict with ``eval_slice`` set -- never mutate the input row."""
    tagged = dict(record)
    tagged["eval_slice"] = slice_name
    return tagged


def assert_has_statement_and_answer(records_by_uid: dict) -> None:
    """Hard-fail if any record is missing a non-blank statement/answer.

    Takes a flat ``{uid: record}`` mapping so it can be called once over
    the union of all three slices, or independently in a test with a
    hand-built mapping.
    """
    missing = [
        uid
        for uid, r in records_by_uid.items()
        if _is_blank(r.get("statement")) or _is_blank(r.get("answer"))
    ]
    if missing:
        missing.sort()
        shown = ", ".join(missing[:10])
        more = "" if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
        raise EvalSetError(
            f"{len(missing)} eval-set record(s) missing statement/answer: "
            f"{shown}{more}"
        )


def assert_no_leakage(records_by_uid: dict, eval_papers) -> None:
    """Hard-fail if any record in ``records_by_uid`` resolves to an eval paper.

    Called on the assembled train set right before writing
    ``train_uids.txt``. The bucketing logic in :func:`build` already
    routes eval-paper records away from ``train_records`` structurally
    (an eval-paper record never reaches the ``else`` branch that adds to
    train), which means normal operation cannot trip this -- it is
    defense-in-depth against a future refactor breaking that structural
    guarantee, so it is re-checked explicitly rather than trusted
    implicitly. Tests exercise it directly with a planted violation.

    ``eval_papers`` may be any container supporting ``in`` (list or set).
    """
    eval_papers_set = eval_papers if isinstance(eval_papers, set) else set(eval_papers)
    leaked = [
        (uid, r.get("arxiv_id"))
        for uid, r in records_by_uid.items()
        if r.get("arxiv_id") in eval_papers_set
    ]
    if leaked:
        leaked.sort(key=lambda pair: pair[0])
        shown = ", ".join(f"{uid}(arxiv_id={arxiv_id})" for uid, arxiv_id in leaked[:10])
        more = "" if len(leaked) <= 10 else f" (+{len(leaked) - 10} more)"
        raise EvalSetError(
            f"LEAKAGE GUARD TRIPPED: {len(leaked)} train uid(s) resolve to "
            f"an eval paper's arxiv_id -- this must never happen (paper-"
            f"level split invariant, docs/eval_harness_design.md). First "
            f"offender(s): {shown}{more}. Refusing to write train_uids.txt "
            "or eval_set.jsonl."
        )


def build(
    *,
    split_path: Path,
    tier_output_paths: list,
    output_dir: Path,
    expected_split_sha16: str = EXPECTED_SPLIT_SHA256_16,
) -> BuildResult:
    """Core build logic, used by both the CLI and tests.

    ``expected_split_sha16`` defaults to the one true production pin;
    tests targeting a small synthetic split pass their own fixture's
    hash explicitly (see module docstring) -- the CLI never exposes an
    override, so production usage cannot accidentally bypass the pin.
    """
    if not tier_output_paths:
        raise EvalSetError("--tier-outputs must list at least one pass_at_k.jsonl path; none were given")

    tier_output_paths = [Path(p) for p in tier_output_paths]
    missing_tiers = [p for p in tier_output_paths if not p.exists()]
    if missing_tiers:
        raise FileNotFoundError(
            "tier-output file(s) not found (cascade incomplete?): "
            + ", ".join(str(p) for p in missing_tiers)
            + ". Pass only --tier-outputs paths that currently exist on "
            "disk -- rerun this tool once more tiers land (see "
            "out/remote_rescore/*/pass_at_k.jsonl)."
        )

    split = load_split(split_path, expected_split_sha16)
    eval_papers = set(split["eval_papers"])

    warnings: list = []
    seen_uids: dict = {}
    eval_band: dict = {}
    anchor_solved: dict = {}
    anchor_fail: dict = {}
    train_records: dict = {}

    for tier_path in tier_output_paths:
        rows_seen = 0
        rows_with_label = 0
        for record in _iter_jsonl(tier_path):
            rows_seen += 1
            uid = record.get("uid")
            if not uid:
                raise EvalSetError(f"{tier_path}: record missing 'uid'")

            if uid in seen_uids:
                warnings.append(
                    f"duplicate uid {uid!r} seen again in {tier_path} "
                    f"(first seen in {seen_uids[uid]}) -- keeping first occurrence"
                )
                continue
            seen_uids[uid] = tier_path

            label = record.get("label")
            if label is not None:
                rows_with_label += 1
            arxiv_id = record.get("arxiv_id")
            is_eval_paper = arxiv_id is not None and arxiv_id in eval_papers

            if is_eval_paper:
                # Any eval-paper record is excluded from training,
                # regardless of label -- so there is no "else: train"
                # branch here. Only the three qualifying shapes below
                # become eval-set members; everything else (misdirection,
                # drop, near-miss solved/collapse) is intentionally inert.
                if label == LABEL_BAND:
                    eval_band[uid] = _tag(record, EVAL_SLICE_BAND)
                elif label == LABEL_SOLVED and _is_perfect_solve(record):
                    anchor_solved[uid] = _tag(record, EVAL_SLICE_ANCHOR_SOLVED)
                elif label == LABEL_COLLAPSE and _is_total_collapse(record):
                    anchor_fail[uid] = _tag(record, EVAL_SLICE_ANCHOR_FAIL)
            else:
                if label == LABEL_BAND:
                    train_records[uid] = record

        if rows_seen > 0 and rows_with_label == 0:
            warnings.append(
                f"{tier_path}: no record had a 'label' field -- is this a "
                "pass_at_k_input.jsonl (pre-scoring) rather than a "
                "pass_at_k.jsonl (post-scoring) file?"
            )

    # Defense-in-depth: re-verify the leakage invariant explicitly before
    # writing anything (see assert_no_leakage's docstring for why this is
    # not dead code).
    assert_no_leakage(train_records, eval_papers)

    all_eval_records = {}
    all_eval_records.update(eval_band)
    all_eval_records.update(anchor_solved)
    all_eval_records.update(anchor_fail)
    assert_has_statement_and_answer(all_eval_records)

    if len(eval_band) < UNDERPOWERED_HINT_N:
        warnings.append(
            f"eval-band has only {len(eval_band)} record(s) (< {UNDERPOWERED_HINT_N}). "
            "Per docs/eval_harness_design.md's Open Items, consider widening "
            "eval-band to include eval-paper 7/8 records and re-stating "
            "power -- this tool does not do so automatically; report.py "
            "will also print an UNDERPOWERED warning at report time."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    eval_set_path = output_dir / "eval_set.jsonl"
    train_uids_path = output_dir / "train_uids.txt"

    ordered_eval_records = (
        [eval_band[u] for u in sorted(eval_band)]
        + [anchor_solved[u] for u in sorted(anchor_solved)]
        + [anchor_fail[u] for u in sorted(anchor_fail)]
    )
    with eval_set_path.open("w", encoding="utf-8") as fh:
        for record in ordered_eval_records:
            fh.write(json.dumps(record) + "\n")

    with train_uids_path.open("w", encoding="utf-8") as fh:
        for uid in sorted(train_records):
            fh.write(uid + "\n")

    counts = {
        "eval_papers_n": len(eval_papers),
        "tier_files": len(tier_output_paths),
        "eval_band": len(eval_band),
        "anchor_solved": len(anchor_solved),
        "anchor_fail": len(anchor_fail),
        "eval_set_total": len(ordered_eval_records),
        "train_band_total": len(train_records),
        "duplicate_uids": sum(1 for w in warnings if w.startswith("duplicate uid")),
    }
    return BuildResult(
        eval_set_path=eval_set_path,
        train_uids_path=train_uids_path,
        counts=counts,
        warnings=warnings,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evalharness-build-set",
        description=(
            "Build the frozen LoRA eval set (eval-band + anchors) from the "
            "paper split and the remote-rescore cascade's pass_at_k outputs."
        ),
    )
    p.add_argument(
        "--split",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help=f"Path to the frozen eval_paper_split.json (default: {DEFAULT_SPLIT_PATH}).",
    )
    p.add_argument(
        "--tier-outputs",
        type=Path,
        nargs="+",
        required=True,
        metavar="PASS_AT_K_JSONL",
        help=(
            "One or more out/remote_rescore/tier*/pass_at_k.jsonl paths. "
            "Tiers may be incomplete on disk at any given time -- pass "
            "only the ones that currently exist; rerun as more land."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where eval_set.jsonl and train_uids.txt are written (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        result = build(
            split_path=args.split,
            tier_output_paths=args.tier_outputs,
            output_dir=args.output_dir,
        )
    except (EvalSetError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "stage": "build_eval_set",
        "split": str(args.split),
        "split_sha256_16": EXPECTED_SPLIT_SHA256_16,
        "tier_outputs": [str(p) for p in args.tier_outputs],
        "counts": result.counts,
        "warnings": result.warnings,
        "outputs": {
            "eval_set": str(result.eval_set_path),
            "train_uids": str(result.train_uids_path),
        },
    }
    print(json.dumps(summary, indent=2))
    for w in result.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
