"""Paired greedy diff, exact McNemar, anchor drift, and the markdown report.

Reads ``eval_set.jsonl`` (for ``eval_slice`` membership) plus the two
greedy-scored files ``run_eval.py`` produces (``baseline_greedy.jsonl``,
``post_greedy.jsonl``), and optionally the k=8 x3 secondary files.
Writes one markdown report to stdout AND to ``--output``.

Statistics, deliberately stdlib-only (no scipy, per repo policy):

  * ``mcnemar_exact`` -- exact two-sided McNemar p-value on the
    discordant pair counts (b, c), computed from the symmetric
    Binomial(n=b+c, p=0.5) distribution via ``math.comb``. This is the
    standard closed-form exact McNemar test (equivalent to the
    "minlike" two-sided exact binomial test for p=0.5, since the
    binomial is symmetric there): p = min(1, 2 * P(X <= min(b, c))).
  * ``wald_ci_paired_diff`` -- a normal-approximation (Wald) 95% CI on
    the paired difference in solve rate. This is an approximation, not
    an exact interval -- exact paired-proportion CIs require an
    iterative search with no closed form, which is out of scope for a
    stdlib-only implementation. Documented as approximate wherever it
    is surfaced.

Never blends the k=8 secondary into the greedy headline -- the design
doc is explicit that the two are reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EVAL_SLICE_BAND = "eval_band"
EVAL_SLICE_ANCHOR_SOLVED = "anchor_solved"
EVAL_SLICE_ANCHOR_FAIL = "anchor_fail"

# docs/eval_harness_design.md: "Under ~25 records, say underpowered out loud."
UNDERPOWERED_THRESHOLD = 25

Z_95 = 1.96  # two-sided 95% Wald critical value


class ReportError(ValueError):
    """Raised on malformed/missing report inputs -- not a statistical result."""


# --- loading -----------------------------------------------------------------


def _load_jsonl(path: Path) -> dict:
    """uid -> row. Raises FileNotFoundError / ReportError on bad input."""
    if not path.exists():
        raise FileNotFoundError(f"report input not found: {path}")
    out = {}
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("uid")
            if uid is None:
                raise ReportError(f"{path}:{lineno}: row missing uid")
            out[uid] = row
    return out


def load_eval_set(path: Path) -> dict:
    return _load_jsonl(path)


def slice_uids(eval_set: dict, slice_name: str) -> list:
    return sorted(uid for uid, row in eval_set.items() if row.get("eval_slice") == slice_name)


def _solved(row: Optional[dict]) -> Optional[bool]:
    """True/False from a greedy-scored row; None if the uid is absent/unscored."""
    if row is None:
        return None
    n_correct = row.get("n_correct")
    if n_correct is None:
        return None
    return n_correct >= 1


# --- paired 2x2 table ----------------------------------------------------------


@dataclass
class PairedTable:
    """Generic 2x2 paired-greedy table over some uid set.

    a = both solved, b = base-only ("regression" when read on an anchor-
    solved slice), c = tuned-only ("the gain" on eval-band; "contamination"
    when read on an anchor-fail slice), d = neither solved.
    """

    n_pairs: int
    a: int
    b: int
    c: int
    d: int
    base_solved_n: int
    tuned_solved_n: int
    missing_uids: list = field(default_factory=list)


def paired_table(uids, base_rows: dict, tuned_rows: dict) -> PairedTable:
    a = b = c = d = 0
    missing = []
    for uid in uids:
        bs = _solved(base_rows.get(uid))
        ts = _solved(tuned_rows.get(uid))
        if bs is None or ts is None:
            missing.append(uid)
            continue
        if bs and ts:
            a += 1
        elif bs and not ts:
            b += 1
        elif not bs and ts:
            c += 1
        else:
            d += 1
    n_pairs = a + b + c + d
    return PairedTable(
        n_pairs=n_pairs,
        a=a,
        b=b,
        c=c,
        d=d,
        base_solved_n=a + b,
        tuned_solved_n=a + c,
        missing_uids=missing,
    )


# --- statistics ----------------------------------------------------------------


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on discordant pair counts (b, c).

    n = b + c discordant pairs, each Bernoulli(0.5) under the null of no
    directional effect. p = min(1, 2 * P(X <= min(b, c))) for X ~
    Binomial(n, 0.5). With zero discordant pairs the data carries no
    directional evidence either way, so p = 1.0 by convention.
    """
    if b < 0 or c < 0:
        raise ValueError(f"discordant pair counts must be >= 0, got b={b}, c={c}")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2 * tail / (2**n))


def wald_ci_paired_diff(b: int, c: int, n_pairs: int, z: float = Z_95) -> tuple:
    """Normal-approximation 95% CI for the paired difference in solve rate.

    diff = (c - b) / n_pairs (positive = tuned solves more than base).
    SE follows the standard paired-proportions variance (Fleiss):
    sqrt(b + c - (c - b)^2 / n_pairs) / n_pairs. Approximate, not exact
    -- see module docstring.
    """
    if n_pairs <= 0:
        return (0.0, 0.0)
    diff = (c - b) / n_pairs
    var = (b + c) - ((c - b) ** 2) / n_pairs
    var = max(var, 0.0)
    se = math.sqrt(var) / n_pairs
    return (diff - z * se, diff + z * se)


# --- secondary (distributional, informational only) ---------------------------


def secondary_distribution(rep_paths: list, uids) -> dict:
    """Per-repeat mean n_correct across ``uids``; skips repeats with no scored rows."""
    uid_set = set(uids)
    per_repeat_mean = []
    for path in rep_paths:
        rows = _load_jsonl(Path(path))
        vals = [rows[u]["n_correct"] for u in uid_set if u in rows and rows[u].get("n_correct") is not None]
        per_repeat_mean.append((sum(vals) / len(vals)) if vals else None)
    observed = [m for m in per_repeat_mean if m is not None]
    return {
        "n_repeats": len(rep_paths),
        "per_repeat_mean_n_correct": per_repeat_mean,
        "overall_mean_n_correct": (sum(observed) / len(observed)) if observed else None,
    }


# --- markdown rendering ---------------------------------------------------------


def _pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.1f}%" if total else "n/a"


def _anchor_section(title: str, table: PairedTable, *, want_solved: bool, red_flag_label: str) -> list:
    lines = [f"### {title}", ""]
    lines.append(f"- n = {table.n_pairs}" + (f" ({len(table.missing_uids)} uid(s) unscored, excluded)" if table.missing_uids else ""))
    lines.append(f"- base solved: {table.base_solved_n} / {table.n_pairs} ({_pct(table.base_solved_n, table.n_pairs)})")
    lines.append(f"- tuned solved: {table.tuned_solved_n} / {table.n_pairs} ({_pct(table.tuned_solved_n, table.n_pairs)})")
    red_flag_n = table.b if want_solved else table.c
    stayed_n = table.a if want_solved else table.d
    lines.append(f"- stayed {'solved' if want_solved else 'failed'}: {stayed_n} / {table.n_pairs}")
    marker = " **(RED FLAG)**" if red_flag_n else ""
    lines.append(f"- {red_flag_label}: {red_flag_n}{marker}")
    lines.append("")
    return lines


def render_markdown(
    *,
    generated_at: str,
    eval_band_table: PairedTable,
    mcnemar_p: float,
    ci: tuple,
    anchor_solved_table: Optional[PairedTable],
    anchor_fail_table: Optional[PairedTable],
    secondary_base: Optional[dict],
    secondary_tuned: Optional[dict],
    baseline_path: str,
    post_path: str,
    eval_set_path: str,
) -> str:
    underpowered = eval_band_table.n_pairs < UNDERPOWERED_THRESHOLD

    lines = []
    lines.append("# LoRA Eval Harness Report")
    lines.append("")
    lines.append(f"Generated: {generated_at}")
    lines.append("")
    lines.append(f"- eval set: `{eval_set_path}`")
    lines.append(f"- baseline (base, greedy): `{baseline_path}`")
    lines.append(f"- post (tuned, greedy): `{post_path}`")
    lines.append("")
    lines.append("## Headline -- eval-band greedy pass@1")
    lines.append("")
    if underpowered:
        lines.append(
            f"> **UNDERPOWERED**: eval-band has {eval_band_table.n_pairs} scored "
            f"record(s), below the design doc's {UNDERPOWERED_THRESHOLD}-record "
            "power floor. Treat any delta below as a signal to investigate, "
            "not a claim."
        )
        lines.append("")
    delta_n = eval_band_table.tuned_solved_n - eval_band_table.base_solved_n
    delta_pp = (
        100.0 * (eval_band_table.tuned_solved_n - eval_band_table.base_solved_n) / eval_band_table.n_pairs
        if eval_band_table.n_pairs
        else 0.0
    )
    lines.append(f"- n (eval-band, paired) = {eval_band_table.n_pairs}")
    if eval_band_table.missing_uids:
        lines.append(f"  - {len(eval_band_table.missing_uids)} eval-band uid(s) unscored in one or both runs, excluded from the pair count")
    lines.append(f"- base solved: {eval_band_table.base_solved_n} / {eval_band_table.n_pairs} ({_pct(eval_band_table.base_solved_n, eval_band_table.n_pairs)})")
    lines.append(f"- tuned solved: {eval_band_table.tuned_solved_n} / {eval_band_table.n_pairs} ({_pct(eval_band_table.tuned_solved_n, eval_band_table.n_pairs)})")
    lines.append(f"- delta solved: {delta_n:+d} ({delta_pp:+.1f}pp)")
    lines.append(f"- discordant pairs: b (base-only correct) = {eval_band_table.b}, c (tuned-only correct) = {eval_band_table.c}")
    lines.append(f"- exact McNemar p = {mcnemar_p:.4g}")
    lines.append(f"- 95% CI (normal approximation, paired) on delta: [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    lines.append("")
    lines.append("## Anchor drift")
    lines.append("")
    lines.append(
        "Anchors are eval-paper records at the extremes (8/8 solved, 0/8 "
        "failed) in the original remote rescore. They are sanity checks, "
        "not the headline: anchor-solved must STAY solved (else "
        "catastrophic forgetting); anchor-fail must STAY failed (else "
        "memorization/contamination)."
    )
    lines.append("")
    if anchor_solved_table is not None:
        lines += _anchor_section(
            "anchor-solved (must stay solved)",
            anchor_solved_table,
            want_solved=True,
            red_flag_label="regressed (base solved, tuned did not)",
        )
    else:
        lines.append("### anchor-solved (must stay solved)")
        lines.append("")
        lines.append("(no anchor-solved records in this eval set)")
        lines.append("")
    if anchor_fail_table is not None:
        lines += _anchor_section(
            "anchor-fail (must stay failed)",
            anchor_fail_table,
            want_solved=False,
            red_flag_label="contaminated (base failed, tuned solved)",
        )
    else:
        lines.append("### anchor-fail (must stay failed)")
        lines.append("")
        lines.append("(no anchor-fail records in this eval set)")
        lines.append("")
    lines.append("## Secondary -- k=8, temperature=0.7, x3 (distributional, informational only)")
    lines.append("")
    lines.append("Never blended into the headline above -- catches probability-mass shifts greedy decoding misses.")
    lines.append("")
    if secondary_base is None or secondary_tuned is None:
        lines.append("(secondary distributional comparison not provided)")
    else:
        lines.append("| repeat | base mean n_correct | tuned mean n_correct |")
        lines.append("|---|---|---|")
        n_reps = max(secondary_base["n_repeats"], secondary_tuned["n_repeats"])
        for i in range(n_reps):
            bm = secondary_base["per_repeat_mean_n_correct"][i] if i < len(secondary_base["per_repeat_mean_n_correct"]) else None
            tm = secondary_tuned["per_repeat_mean_n_correct"][i] if i < len(secondary_tuned["per_repeat_mean_n_correct"]) else None
            lines.append(f"| {i} | {bm if bm is not None else 'n/a'} | {tm if tm is not None else 'n/a'} |")
        lines.append("")
        ob = secondary_base["overall_mean_n_correct"]
        ot = secondary_tuned["overall_mean_n_correct"]
        lines.append(f"- overall mean n_correct: base={ob if ob is not None else 'n/a'}, tuned={ot if ot is not None else 'n/a'}")
    lines.append("")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evalharness-report",
        description="Paired greedy diff, exact McNemar, anchor drift, and markdown report.",
    )
    p.add_argument("--eval-set", type=Path, required=True, help="eval_set.jsonl (build_eval_set.py output).")
    p.add_argument("--baseline", type=Path, required=True, help="baseline_greedy.jsonl (run_eval.py --model-base output).")
    p.add_argument("--post", type=Path, required=True, help="post_greedy.jsonl (run_eval.py --model-tuned output).")
    p.add_argument("--secondary-base", type=Path, nargs="+", default=None, help="Base model's k=8 x3 secondary pass_at_k.jsonl files, one per repeat.")
    p.add_argument("--secondary-post", type=Path, nargs="+", default=None, help="Tuned model's k=8 x3 secondary pass_at_k.jsonl files, one per repeat.")
    p.add_argument("--output", type=Path, required=True, help="Markdown report destination (also printed to stdout).")
    return p


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_report(
    *,
    eval_set_path: Path,
    baseline_path: Path,
    post_path: Path,
    secondary_base_paths: Optional[list] = None,
    secondary_post_paths: Optional[list] = None,
    generated_at: Optional[str] = None,
) -> str:
    """Build the markdown report as a string. Pure function -- no I/O beyond reads.

    ``generated_at`` defaults to the current UTC time; tests pass a fixed
    string so golden-file comparisons stay byte-stable.
    """
    eval_set = load_eval_set(eval_set_path)
    base_rows = _load_jsonl(baseline_path)
    tuned_rows = _load_jsonl(post_path)

    eval_band_uids = slice_uids(eval_set, EVAL_SLICE_BAND)
    anchor_solved_uids = slice_uids(eval_set, EVAL_SLICE_ANCHOR_SOLVED)
    anchor_fail_uids = slice_uids(eval_set, EVAL_SLICE_ANCHOR_FAIL)

    if not eval_band_uids:
        raise ReportError(f"{eval_set_path} has zero eval_slice={EVAL_SLICE_BAND!r} records -- nothing to report")

    eval_band_table = paired_table(eval_band_uids, base_rows, tuned_rows)
    anchor_solved_table = paired_table(anchor_solved_uids, base_rows, tuned_rows) if anchor_solved_uids else None
    anchor_fail_table = paired_table(anchor_fail_uids, base_rows, tuned_rows) if anchor_fail_uids else None

    p_value = mcnemar_exact(eval_band_table.b, eval_band_table.c)
    ci = wald_ci_paired_diff(eval_band_table.b, eval_band_table.c, eval_band_table.n_pairs)

    secondary_base = secondary_tuned = None
    if secondary_base_paths and secondary_post_paths:
        secondary_base = secondary_distribution(secondary_base_paths, eval_band_uids)
        secondary_tuned = secondary_distribution(secondary_post_paths, eval_band_uids)

    return render_markdown(
        generated_at=generated_at or _now_utc(),
        eval_band_table=eval_band_table,
        mcnemar_p=p_value,
        ci=ci,
        anchor_solved_table=anchor_solved_table,
        anchor_fail_table=anchor_fail_table,
        secondary_base=secondary_base,
        secondary_tuned=secondary_tuned,
        # Basename only, not the full path: keeps the report portable
        # (shareable/archivable independent of the machine/checkout it was
        # generated on) and keeps golden-file tests stable regardless of
        # whether callers pass relative or absolute paths.
        baseline_path=Path(baseline_path).name,
        post_path=Path(post_path).name,
        eval_set_path=Path(eval_set_path).name,
    )


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        md = generate_report(
            eval_set_path=args.eval_set,
            baseline_path=args.baseline,
            post_path=args.post,
            secondary_base_paths=args.secondary_base,
            secondary_post_paths=args.secondary_post,
        )
    except (FileNotFoundError, ReportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(md)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
