"""N-way cross-combo comparison.

Joins N normalised verdict streams on ``uid`` and classifies the
agreement pattern across the fleet. Emits per-record comparison rows
and a human-readable markdown rollup that leads with outcome → counts
→ pairwise agreement matrix → top disagreements.

Agreement buckets generalise the 2-poser scheme:

    unanimous_pass    — every present combo says well_posed
    unanimous_fail    — every present combo says ill_posed
    unanimous_defer   — every present combo says defer
    split             — combos disagree (the catch-all for mixed)
    has_missing       — at least one combo did not return a verdict

Pairwise Cohen's kappa is reported for every combo pair; the full
matrix is included in the markdown report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Optional

from icepick.processing.poser.base import (
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserVerdict,
)

UNANIMOUS_PASS = "unanimous_pass"
UNANIMOUS_FAIL = "unanimous_fail"
UNANIMOUS_DEFER = "unanimous_defer"
SPLIT = "split"
HAS_MISSING = "has_missing"
AGREEMENT_BUCKETS = (UNANIMOUS_PASS, UNANIMOUS_FAIL, UNANIMOUS_DEFER, SPLIT, HAS_MISSING)


@dataclass
class Comparison:
    rows: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    total: int = 0
    pairwise_kappa: dict = field(default_factory=dict)  # {(combo_a, combo_b): kappa | None}
    top_disagreements: list = field(default_factory=list)


def compare_verdicts(
    *,
    verdicts_by_combo: dict,        # {combo_key: list[PoserVerdict]}
    statements_by_uid: dict,
) -> Comparison:
    combo_keys = list(verdicts_by_combo.keys())
    by_combo_by_uid: dict = {
        ck: {v.uid: v for v in verdicts_by_combo[ck]} for ck in combo_keys
    }
    all_uids = sorted(set().union(*[set(d) for d in by_combo_by_uid.values()]))

    rows: list = []
    counts: dict = {b: 0 for b in AGREEMENT_BUCKETS}

    for uid in all_uids:
        per_combo: dict = {}
        statement = statements_by_uid.get(uid, "") or ""
        preview = statement[:120] + ("..." if len(statement) > 120 else "")
        source = ""
        for ck in combo_keys:
            v = by_combo_by_uid[ck].get(uid)
            per_combo[ck] = _verdict_summary(v)
            if v and not source:
                source = v.source

        present = [s["verdict_status"] for s in per_combo.values() if s is not None]
        if len(present) < len(combo_keys):
            bucket = HAS_MISSING
        elif all(s == STATUS_WELL_POSED for s in present):
            bucket = UNANIMOUS_PASS
        elif all(s == STATUS_ILL_POSED for s in present):
            bucket = UNANIMOUS_FAIL
        elif all(s == STATUS_DEFER for s in present):
            bucket = UNANIMOUS_DEFER
        else:
            bucket = SPLIT
        counts[bucket] += 1
        rows.append(
            {
                "uid": uid,
                "source": source,
                "statement_preview": preview,
                "per_combo": per_combo,
                "agreement": bucket,
            }
        )

    pairwise_kappa: dict = {}
    for a, b in combinations(combo_keys, 2):
        pairwise_kappa[(a, b)] = _cohen_kappa(by_combo_by_uid[a], by_combo_by_uid[b])

    top_disagreements = [r for r in rows if r["agreement"] in (SPLIT, HAS_MISSING)][:20]

    return Comparison(
        rows=rows,
        counts=counts,
        total=len(rows),
        pairwise_kappa=pairwise_kappa,
        top_disagreements=top_disagreements,
    )


def write_comparison_report(comparison: Comparison, path: Path, *, combo_keys: list) -> None:
    """Markdown rollup: outcome → counts → pairwise kappa → top disagreements."""
    lines: list = []
    lines.append("# Wellposed fleet comparison report")
    lines.append("")
    lines.append(f"Fleet size: **{len(combo_keys)}** — {', '.join(f'`{k}`' for k in combo_keys)}")
    lines.append(f"Total records compared: **{comparison.total}**")
    lines.append("")
    lines.append("## Agreement counts")
    lines.append("")
    lines.append("| bucket | count | share |")
    lines.append("| ------ | ----- | ----- |")
    total = max(comparison.total, 1)
    for bucket in AGREEMENT_BUCKETS:
        n = comparison.counts.get(bucket, 0)
        share = n / total
        lines.append(f"| `{bucket}` | {n} | {share:.1%} |")
    lines.append("")

    if comparison.pairwise_kappa:
        lines.append("## Pairwise Cohen's kappa")
        lines.append("")
        lines.append("| pair | kappa |")
        lines.append("| ---- | ----- |")
        for (a, b), kappa in comparison.pairwise_kappa.items():
            kstr = f"{kappa:.3f}" if kappa is not None else "—"
            lines.append(f"| `{a}` vs `{b}` | {kstr} |")
        lines.append("")

    if comparison.top_disagreements:
        lines.append(f"## Top {len(comparison.top_disagreements)} disagreements / missing")
        lines.append("")
        header = "| uid | source |" + "".join(f" {k} |" for k in combo_keys) + " preview |"
        sep = "| --- | ------ |" + "".join(" --- |" for _ in combo_keys) + " ------- |"
        lines.append(header)
        lines.append(sep)
        for row in comparison.top_disagreements:
            cells = []
            for ck in combo_keys:
                summary = row["per_combo"].get(ck)
                cells.append(summary["verdict_status"] if summary else "—")
            cell_str = "".join(f" `{c}` |" for c in cells)
            lines.append(
                "| `{uid}` | {src} |{cells} {pv} |".format(
                    uid=row["uid"][:12],
                    src=row.get("source") or "",
                    cells=cell_str,
                    pv=row.get("statement_preview", "").replace("|", "\\|"),
                )
            )
        lines.append("")
    lines.append("Full per-record comparison: `comparison.jsonl`.")
    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _verdict_summary(v: Optional[PoserVerdict]) -> Optional[dict]:
    if v is None:
        return None
    return {
        "verdict_status": v.verdict_status,
        "verdict_score": v.verdict_score,
        "poser_model": v.poser_model,
    }


def _cohen_kappa(by_a: dict, by_b: dict) -> Optional[float]:
    """Cohen's kappa on the canonical 4-value space, pure stdlib."""
    shared_uids = set(by_a) & set(by_b)
    if not shared_uids:
        return None
    labels = (STATUS_WELL_POSED, STATUS_ILL_POSED, STATUS_DEFER, STATUS_ERROR)
    label_idx = {l: i for i, l in enumerate(labels)}
    n = len(shared_uids)
    confusion = [[0] * len(labels) for _ in labels]
    for uid in shared_uids:
        i = label_idx[by_a[uid].verdict_status]
        j = label_idx[by_b[uid].verdict_status]
        confusion[i][j] += 1
    observed = sum(confusion[i][i] for i in range(len(labels))) / n
    row_totals = [sum(row) for row in confusion]
    col_totals = [sum(confusion[i][j] for i in range(len(labels))) for j in range(len(labels))]
    expected = sum((row_totals[i] * col_totals[i]) for i in range(len(labels))) / (n * n)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)
