"""Wellposed-stage runner — fleet of (build, provider) combinations in parallel.

Flow:

  1. Read pass@k records (JSONL).
  2. Inject ``uid`` on every record.
  3. For each combo in ``cfg.combos`` (e.g. ``claude:anthropic``,
     ``codex:openai``), build → run → normalise. Subprocess fan-out is
     parallel by default; ``serialize_fleet=true`` falls back to
     sequential for shared-rate-limit scenarios.
  4. Write one normalised JSONL per combo to ``out/wellposed/{slug}_normalised.jsonl``.
  5. If more than one combo ran, also write ``comparison.jsonl`` +
     ``comparison_report.md`` and assemble a combined gate-input file
     under ``comparison_policy``. With one combo the gate input is just
     that combo's normalised file.
  6. Write ``run_manifest.json`` recording mode, fleet, counts, paths,
     timings, and config echo.

The runner is deliberately a thin coordinator. All subprocess details
live inside the adapters; all comparison logic lives in
``comparator.py``. This module owns only orchestration and IO.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from icepick.processing.poser.base import (
    STATUS_DEFER,
    STATUS_ERROR,
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserVerdict,
    inject_uid,
)
from icepick.processing.poser.claude_adapter import ClaudePoserAdapter
from icepick.processing.poser.codex_adapter import CodexPoserAdapter
from icepick.processing.poser.comparator import (
    compare_verdicts,
    write_comparison_report,
)
from icepick.processing.poser.config import (
    BUILD_CLAUDE,
    BUILD_CODEX,
    POLICY_INTERSECT,
    POLICY_MAJORITY,
    POLICY_PREFER,
    POLICY_UNION,
    Combo,
    WellposedConfig,
)


@dataclass
class RunOutcome:
    """What the runner returns once a wellposed run completes."""

    manifest_path: Path
    normalised_paths: dict          # {combo_key: Path}
    gate_input_path: Path           # combined verdicts file (single combo OR policy-combined)
    passed_records_path: Path       # original record dicts for uids that passed the gate
    comparison_path: Optional[Path] # only when fleet size > 1
    counts: dict                    # {'well_posed': N, 'ill_posed': N, ...}


def run(
    *,
    cfg: WellposedConfig,
    records: Iterable[dict],
    adapter_overrides: Optional[dict] = None,
) -> RunOutcome:
    """Execute the wellposed stage end-to-end. Returns a RunOutcome.

    ``adapter_overrides`` lets tests inject mocked adapters keyed by
    BUILD name (``'claude'``, ``'codex'``). Production callers leave it
    None and get the real subprocess-driven adapters.
    """
    cfg.validate()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = inject_uid(list(records))
    input_uids = [r["uid"] for r in prepared]
    pass_at_k_copy = output_dir / "poser_input.jsonl"
    with pass_at_k_copy.open("w", encoding="utf-8") as fh:
        for record in prepared:
            fh.write(json.dumps(record) + "\n")

    adapters_by_build = _resolve_adapters(adapter_overrides)

    # Plan one PoserRequest per combo.
    invocations: list = []
    for combo in cfg.combos:
        adapter = adapters_by_build[combo.build]
        req = adapter.plan(prepared, cfg, combo, output_dir)
        invocations.append((combo, adapter, req))

    # Fan out subprocess execution.
    if len(invocations) > 1 and not cfg.serialize_fleet:
        with ThreadPoolExecutor(max_workers=len(invocations)) as ex:
            run_results = list(
                ex.map(
                    lambda inv: (inv[0], inv[1], inv[2], inv[1].run(inv[2])),
                    invocations,
                )
            )
    else:
        run_results = [
            (combo, adapter, req, adapter.run(req))
            for combo, adapter, req in invocations
        ]

    # Normalise every adapter's output into canonical PoserVerdict streams.
    normalised: dict = {}            # combo_key -> list[PoserVerdict]
    normalised_paths: dict = {}      # combo_key -> Path
    for combo, adapter, req, run_result in run_results:
        verdicts = adapter.normalise(run_result.output_path, input_uids, combo=combo)
        norm_path = output_dir / f"{combo.slug()}_normalised.jsonl"
        _write_verdicts(verdicts, norm_path)
        normalised[combo.key()] = verdicts
        normalised_paths[combo.key()] = norm_path

    # If the fleet has more than one combo, write a comparison and
    # assemble the combined gate-input file under the chosen policy.
    comparison_path: Optional[Path] = None
    if len(cfg.combos) > 1:
        comparison = compare_verdicts(
            verdicts_by_combo={c.key(): normalised[c.key()] for c in cfg.combos},
            statements_by_uid={r["uid"]: r.get("statement", "") for r in prepared},
        )
        comparison_path = output_dir / "comparison.jsonl"
        with comparison_path.open("w", encoding="utf-8") as fh:
            for row in comparison.rows:
                fh.write(json.dumps(row) + "\n")
        write_comparison_report(
            comparison,
            output_dir / "comparison_report.md",
            combo_keys=[c.key() for c in cfg.combos],
        )
        gate_input_path = _select_gate_input(cfg, output_dir, normalised, normalised_paths)
    else:
        gate_input_path = normalised_paths[cfg.combos[0].key()]

    counts = _count_statuses_in_gate_input(gate_input_path)
    token_usage = _aggregate_token_usage(normalised, cfg=cfg)

    # Write the records (in their original ingest shape) for uids whose
    # combined verdict is well_posed. This is the file downstream consumers
    # treat as the gate's output corpus — mirrors groundtruth's published.jsonl.
    passed_records_path = output_dir / "passed_records.jsonl"
    _write_passed_records(gate_input_path, prepared, passed_records_path)

    manifest = {
        "stage": "wellposed",
        "config": cfg.echo(),
        "inputs": {
            "pass_at_k_copy": str(pass_at_k_copy),
            "record_count": len(prepared),
        },
        "outputs": {
            "normalised_paths": {k: str(v) for k, v in normalised_paths.items()},
            "comparison_path": str(comparison_path) if comparison_path else None,
            "gate_input_path": str(gate_input_path),
            "passed_records_path": str(passed_records_path),
        },
        "token_usage": token_usage,
        "subprocess_runs": [
            {
                "combo": combo.key(),
                "argv": req.argv,
                "exit_code": rr.exit_code,
                "wall_clock_seconds": rr.wall_clock_seconds,
                "stderr_tail": rr.stderr[-2000:] if rr.stderr else "",
            }
            for combo, adapter, req, rr in run_results
        ],
        "counts": counts,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return RunOutcome(
        manifest_path=manifest_path,
        normalised_paths=normalised_paths,
        gate_input_path=gate_input_path,
        passed_records_path=passed_records_path,
        comparison_path=comparison_path,
        counts=counts,
    )


def _write_passed_records(gate_input_path: Path, prepared: list, out_path: Path) -> None:
    """Write the original record dicts for uids whose combined verdict is well_posed.

    The gate-input file is keyed by uid and carries verdicts only; this
    join produces a JSONL of the original records, ready to be consumed
    as a corpus by downstream tooling.
    """
    passing_uids: set = set()
    with gate_input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("verdict_status") == STATUS_WELL_POSED:
                passing_uids.add(row.get("uid"))
    with out_path.open("w", encoding="utf-8") as fh:
        for record in prepared:
            if record.get("uid") in passing_uids:
                fh.write(json.dumps(record) + "\n")


def _aggregate_token_usage(normalised: dict, *, cfg: WellposedConfig) -> dict:
    """Sum verdict_signals.usage across every combo and every record.

    The per-combo breakdown lets operators see "the openai combo cost N
    tokens vs the anthropic combo cost M" — useful when comparing
    providers. The fleet-level totals + estimated_cost mirror the
    groundtruth manifest layout.
    """
    per_combo: dict = {}
    fleet_totals = _zero_usage()
    for combo_key, verdicts in normalised.items():
        combo_totals = _zero_usage()
        for v in verdicts:
            usage = (v.verdict_signals or {}).get("usage") or {}
            if not usage:
                continue
            combo_totals["records_with_usage"] += 1
            for field in ("input_tokens", "output_tokens",
                          "cache_read_input_tokens", "cache_creation_input_tokens"):
                combo_totals[field] += int(usage.get(field) or 0)
        per_combo[combo_key] = combo_totals
        for k in fleet_totals:
            fleet_totals[k] += combo_totals[k]

    out: dict = {"fleet_totals": fleet_totals, "per_combo": per_combo}
    if cfg.cost_per_input_mtok is not None or cfg.cost_per_output_mtok is not None:
        in_rate = cfg.cost_per_input_mtok or 0.0
        out_rate = cfg.cost_per_output_mtok or 0.0
        out["estimated_cost"] = {
            "input_usd": round(fleet_totals["input_tokens"] / 1_000_000 * in_rate, 6),
            "output_usd": round(fleet_totals["output_tokens"] / 1_000_000 * out_rate, 6),
            "total_usd": round(
                (fleet_totals["input_tokens"] / 1_000_000 * in_rate)
                + (fleet_totals["output_tokens"] / 1_000_000 * out_rate),
                6,
            ),
            "is_estimate": True,
            "rates_per_mtok": {"input_usd": in_rate, "output_usd": out_rate},
        }
    return out


def _zero_usage() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "records_with_usage": 0,
    }


def _resolve_adapters(overrides: Optional[dict]) -> dict:
    overrides = overrides or {}
    defaults = {BUILD_CLAUDE: ClaudePoserAdapter(), BUILD_CODEX: CodexPoserAdapter()}
    return {build: overrides.get(build, defaults[build]) for build in defaults}


def _write_verdicts(verdicts: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for v in verdicts:
            fh.write(json.dumps(v.to_jsonl_row()) + "\n")


def _select_gate_input(
    cfg: WellposedConfig,
    output_dir: Path,
    normalised: dict,
    normalised_paths: dict,
) -> Path:
    """Apply ``comparison_policy`` to pick or assemble the gate-input file."""
    policy = cfg.comparison_policy

    if policy.startswith(POLICY_PREFER):
        target = policy[len(POLICY_PREFER):]
        return normalised_paths[target]

    # Combinator policies: write one synthesised JSONL combining all combos.
    all_uids = sorted({v.uid for combo_key in normalised for v in normalised[combo_key]})
    combined_path = output_dir / f"combined_{policy.replace(':','_')}.jsonl"
    combo_keys = [c.key() for c in cfg.combos]
    with combined_path.open("w", encoding="utf-8") as fh:
        for uid in all_uids:
            row = _combine_n(uid, combo_keys, normalised, policy)
            fh.write(json.dumps(row) + "\n")
    return combined_path


def _combine_n(uid: str, combo_keys: list, normalised: dict, policy: str) -> dict:
    """Synthesise one combined verdict row across N combos.

    When admit=False, the combined status carries the most informative
    denial signal: ill_posed > error > defer. This keeps the gate's
    downstream filter (admit iff verdict_status == well_posed) honest
    AND surfaces *why* the record was denied.
    """
    per_combo: dict = {}
    source = ""
    for ck in combo_keys:
        v = _find(uid, normalised[ck])
        per_combo[ck] = v.verdict_status if v else None
        if v and not source:
            source = v.source

    statuses = [s for s in per_combo.values() if s is not None]
    n = len(statuses)
    n_pass = sum(1 for s in statuses if s == STATUS_WELL_POSED)

    if policy == POLICY_INTERSECT:
        admit = n == len(combo_keys) and n_pass == n
    elif policy == POLICY_UNION:
        admit = n_pass >= 1
    elif policy == POLICY_MAJORITY:
        admit = n > 0 and n_pass * 2 > n
    else:
        admit = False  # unreachable for valid policies

    if admit:
        status = STATUS_WELL_POSED
    else:
        for candidate in (STATUS_ILL_POSED, STATUS_ERROR, STATUS_DEFER):
            if candidate in statuses:
                status = candidate
                break
        else:
            status = STATUS_ERROR

    return {
        "uid": uid,
        "source": source,
        "verdict_status": status,
        "verdict_score": 1.0 if admit else 0.0,
        "poser_name": "combined",
        "poser_model": "",
        "verdict_detail": {
            "policy": policy,
            "per_combo": per_combo,
            "pass_count": n_pass,
            "fleet_size": len(combo_keys),
        },
        "verdict_signals": {},
        "raw_payload": {},
    }


def _find(uid: str, verdicts: list):
    for v in verdicts:
        if v.uid == uid:
            return v
    return None


def _count_statuses_in_gate_input(path: Path) -> dict:
    counts: dict = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = row.get("verdict_status", "error")
            counts[status] = counts.get(status, 0) + 1
    return counts
