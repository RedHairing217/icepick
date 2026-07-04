"""3-stage wellposed cascade — sequential single-combo elimination.

The cascade drives a series of single-combo wellposed runs in order. Each
stage runs one combo on the survivors of the previous stage. Records that
any stage rejects (ill_posed, defer, or exhausted-error) are dropped
permanently. The final corpus is the intersection of every stage's
well_posed set.

Compared to the parallel fleet + comparison policy in ``runner.py``, the
cascade trades wall-clock for tokens: later stages only see records that
earlier (cheaper, more permissive) stages already passed. It also gives
each stage a clean unit boundary — one run manifest per stage — so
downstream consumers can attribute cost, timing, and verdicts to a single
(build, provider) combo.

Retries are per-uid, at the cascade layer: when the vendored codex-poser
returns transient network errors, the cascade re-invokes the stage on
just the errored uids up to ``max_retries`` times with exponential backoff
before giving up. The retry runner reuses the standard ``runner.run``
wellposed machinery; retry attempts land in ``<stage_subdir>/retry_N/``
for audit while the merged view sits at the stage subdir top level.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

from icepick.config import ConfigError
from icepick.processing.poser.base import (
    STATUS_ERROR,
    STATUS_WELL_POSED,
    PoserVerdict,
    inject_uid,
)
from icepick.processing.poser.config import (
    EXTRACTED_JUDGE_POLICY_ALWAYS,
    POLICY_INTERSECT,
    Combo,
    PoserSettings,
    WellposedConfig,
    parse_combo,
)
from icepick.processing.poser.runner import run as run_wellposed

# --- default stage list --------------------------------------------------
#
# codex:openai first (cheapest, most permissive) sheds the majority of
# ill-posed records; codex:anthropic then applies the strict formalist
# check; claude:openai provides a semantic sanity pass. Records that all
# three admit form the 3-way unanimous well_posed corpus. Operators can
# override via ``--stages``.
DEFAULT_STAGES: tuple = ("codex:openai", "codex:anthropic", "claude:openai")


@dataclass(frozen=True)
class StageSpec:
    """One position in the cascade — the ordinal and the combo to run."""

    index: int  # 1-based, for manifest + subdir naming
    combo: Combo

    @property
    def label(self) -> str:
        return self.combo.slug()

    @property
    def subdir_name(self) -> str:
        return f"stage_{self.index}_{self.combo.slug()}"


def parse_stages(specs: List[str]) -> List[StageSpec]:
    """Turn ordered 'build:provider' strings into StageSpecs (1-indexed)."""
    return [StageSpec(index=i, combo=parse_combo(spec))
            for i, spec in enumerate(specs, start=1)]


@dataclass
class CascadeConfig:
    """Config for a wellposed cascade.

    ``stages`` is the ordered pipeline. Each stage materialises into a
    single-combo :class:`WellposedConfig` at run time — the cascade never
    duplicates the wellposed field surface, it only decides ORDERING and
    RETRIES.
    """

    stages: List[StageSpec] = field(default_factory=list)
    mode: str = "production"
    output_dir: Path = field(default_factory=lambda: Path("out/wellposed_cascade"))
    anthropic_key_file: Optional[Path] = None
    openai_key_file: Optional[Path] = None
    calibration_sheet: Optional[Path] = None
    enable_judge_tier: bool = True
    judge_samples: int = 3
    judge_uphold: int = 2
    extracted_judge_policy: str = EXTRACTED_JUDGE_POLICY_ALWAYS
    serialize_fleet: bool = False
    # Per-stage retry policy (applies to ANY STATUS_ERROR verdict).
    max_retries: int = 2
    retry_base_delay: float = 2.0
    retry_max_delay: float = 30.0
    cost_per_input_mtok: Optional[float] = None
    cost_per_output_mtok: Optional[float] = None
    claude: PoserSettings = field(
        default_factory=lambda: PoserSettings(cli_path="claude-poser")
    )
    codex: PoserSettings = field(
        default_factory=lambda: PoserSettings(cli_path="codex-poser")
    )

    def validate(self) -> None:
        if not self.stages:
            raise ConfigError("cascade.stages must list at least one stage")
        for i, stage in enumerate(self.stages, start=1):
            if stage.index != i:
                raise ConfigError(
                    f"cascade.stages[{i-1}].index must be {i}, got {stage.index}"
                )
        if self.max_retries < 0:
            raise ConfigError("cascade.max_retries must be >= 0")
        if self.retry_base_delay < 0:
            raise ConfigError("cascade.retry_base_delay must be >= 0")
        if self.retry_max_delay < self.retry_base_delay:
            raise ConfigError(
                "cascade.retry_max_delay must be >= retry_base_delay"
            )
        # Delegate combo/mode/judge-tier invariants to each stage's own
        # WellposedConfig.validate() so the two layers agree byte for byte.
        for stage in self.stages:
            self.stage_wellposed_config(stage).validate()

    def stage_wellposed_config(
        self,
        stage: StageSpec,
        *,
        subdir: Optional[str] = None,
    ) -> WellposedConfig:
        """Materialise a single-combo WellposedConfig for one stage.

        ``subdir`` overrides the default subdir path (used to isolate
        retry attempts under ``<stage>/retry_N/``).
        """
        target_subdir = subdir if subdir is not None else stage.subdir_name
        wp = WellposedConfig(
            combos=[stage.combo],
            mode=self.mode,
            output_dir=Path(self.output_dir) / target_subdir,
            anthropic_key_file=self.anthropic_key_file,
            openai_key_file=self.openai_key_file,
            enable_judge_tier=self.enable_judge_tier,
            judge_samples=self.judge_samples,
            judge_uphold=self.judge_uphold,
            calibration_sheet=self.calibration_sheet,
            comparison_policy=POLICY_INTERSECT,
            extracted_judge_policy=self.extracted_judge_policy,
            serialize_fleet=self.serialize_fleet,
            cost_per_input_mtok=self.cost_per_input_mtok,
            cost_per_output_mtok=self.cost_per_output_mtok,
        )
        wp.claude.cli_path = self.claude.cli_path
        wp.claude.judge_model = self.claude.judge_model
        wp.claude.extra_args = list(self.claude.extra_args)
        wp.codex.cli_path = self.codex.cli_path
        wp.codex.judge_model = self.codex.judge_model
        wp.codex.extra_args = list(self.codex.extra_args)
        return wp

    def echo(self) -> dict:
        return {
            "stages": [
                {"index": s.index, "combo": s.combo.key(), "slug": s.combo.slug()}
                for s in self.stages
            ],
            "mode": self.mode,
            "output_dir": str(self.output_dir),
            "anthropic_key_file": str(self.anthropic_key_file) if self.anthropic_key_file else None,
            "openai_key_file": str(self.openai_key_file) if self.openai_key_file else None,
            "calibration_sheet": str(self.calibration_sheet) if self.calibration_sheet else None,
            "enable_judge_tier": self.enable_judge_tier,
            "judge_samples": self.judge_samples,
            "judge_uphold": self.judge_uphold,
            "extracted_judge_policy": self.extracted_judge_policy,
            "serialize_fleet": self.serialize_fleet,
            "max_retries": self.max_retries,
            "retry_base_delay": self.retry_base_delay,
            "retry_max_delay": self.retry_max_delay,
            "cost_per_input_mtok": self.cost_per_input_mtok,
            "cost_per_output_mtok": self.cost_per_output_mtok,
            "claude": {
                "cli_path": self.claude.cli_path,
                "judge_model": self.claude.judge_model,
                "extra_args": list(self.claude.extra_args),
            },
            "codex": {
                "cli_path": self.codex.cli_path,
                "judge_model": self.codex.judge_model,
                "extra_args": list(self.codex.extra_args),
            },
        }


@dataclass
class CascadeStageOutcome:
    """Per-stage rollup — one entry in :attr:`CascadeOutcome.stages`."""

    stage: StageSpec
    wellposed_manifest_path: Path
    normalised_path: Path
    passed_records_path: Path
    input_uid_count: int
    survivor_uid_count: int
    counts: dict
    wall_clock_seconds: float
    token_usage: dict
    estimated_cost_usd: Optional[float]
    retry_events: List[dict] = field(default_factory=list)


@dataclass
class CascadeOutcome:
    manifest_path: Path
    final_corpus_path: Path
    final_corpus_count: int
    stages: List[CascadeStageOutcome]
    overall_counts: dict
    total_token_usage: dict
    total_estimated_cost_usd: Optional[float]
    total_wall_clock_seconds: float


def run_cascade(
    *,
    cfg: CascadeConfig,
    records: Iterable[dict],
    adapter_overrides: Optional[dict] = None,
    sleep_fn=time.sleep,
) -> CascadeOutcome:
    """Execute the cascade end-to-end.

    ``adapter_overrides`` flows through to each stage's :func:`runner.run`
    call (build-keyed). ``sleep_fn`` is injectable so retry backoff can be
    stubbed in tests.
    """
    cfg.validate()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prepared = inject_uid(list(records))
    initial_count = len(prepared)
    cascade_start = time.monotonic()

    stage_outcomes: List[CascadeStageOutcome] = []
    survivor_counts: List[int] = []
    current_records = prepared

    for stage in cfg.stages:
        stage_wall_start = time.monotonic()
        outcome = _run_stage_with_retries(
            cfg=cfg,
            stage=stage,
            records=current_records,
            adapter_overrides=adapter_overrides,
            sleep_fn=sleep_fn,
        )
        outcome.wall_clock_seconds = round(time.monotonic() - stage_wall_start, 3)
        stage_outcomes.append(outcome)
        current_records = _load_records_jsonl(outcome.passed_records_path)
        survivor_counts.append(len(current_records))

    final_corpus_path = output_dir / "final_corpus.jsonl"
    with final_corpus_path.open("w", encoding="utf-8") as fh:
        for record in current_records:
            fh.write(json.dumps(record) + "\n")
    final_count = len(current_records)

    total_wall = round(time.monotonic() - cascade_start, 3)
    total_tokens = _sum_token_usage(so.token_usage for so in stage_outcomes)
    total_cost = _sum_estimated_cost(cfg, total_tokens)

    overall_counts: dict = {"initial_record_count": initial_count}
    for i, count in enumerate(survivor_counts, start=1):
        overall_counts[f"after_stage_{i}"] = count
    overall_counts["final_corpus_count"] = final_count
    overall_counts["dropped_total"] = initial_count - final_count

    manifest = {
        "stage": "wellposed_cascade",
        "config": cfg.echo(),
        "inputs": {"initial_record_count": initial_count},
        "stages": [_stage_manifest_entry(so) for so in stage_outcomes],
        "overall": {
            **overall_counts,
            "total_token_usage": total_tokens,
            "total_estimated_cost_usd": total_cost,
            "total_wall_clock_seconds": total_wall,
        },
        "outputs": {"final_corpus_path": str(final_corpus_path)},
    }
    manifest_path = output_dir / "cascade_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return CascadeOutcome(
        manifest_path=manifest_path,
        final_corpus_path=final_corpus_path,
        final_corpus_count=final_count,
        stages=stage_outcomes,
        overall_counts=overall_counts,
        total_token_usage=total_tokens,
        total_estimated_cost_usd=total_cost,
        total_wall_clock_seconds=total_wall,
    )


def _run_stage_with_retries(
    *,
    cfg: CascadeConfig,
    stage: StageSpec,
    records: List[dict],
    adapter_overrides: Optional[dict],
    sleep_fn,
) -> CascadeStageOutcome:
    """Drive one stage, retrying transient errors up to ``cfg.max_retries``.

    After all attempts settle, the stage's top-level normalised JSONL and
    passed_records JSONL are (re)written from the merged verdict set. The
    initial attempt's ``run_manifest.json`` remains the canonical stage
    manifest; retry attempts land in nested ``retry_N/`` subdirs.
    """
    combo_key = stage.combo.key()
    all_verdicts: dict = {}     # uid -> latest PoserVerdict (last-write-wins)
    retry_events: List[dict] = []
    attempt_manifests: List[Path] = []
    attempt_token_usages: List[dict] = []

    pending_records = list(records)
    stage_subdir = Path(cfg.output_dir) / stage.subdir_name
    stage_subdir.mkdir(parents=True, exist_ok=True)

    for attempt in range(cfg.max_retries + 1):
        if not pending_records:
            break
        subdir = (
            stage.subdir_name
            if attempt == 0
            else f"{stage.subdir_name}/retry_{attempt}"
        )
        attempt_cfg = cfg.stage_wellposed_config(stage, subdir=subdir)
        run_outcome = run_wellposed(
            cfg=attempt_cfg,
            records=pending_records,
            adapter_overrides=adapter_overrides,
        )
        attempt_manifests.append(run_outcome.manifest_path)
        # Pull token usage recorded by the runner for this attempt.
        try:
            m = json.loads(run_outcome.manifest_path.read_text())
            attempt_token_usages.append(m.get("token_usage", {}))
        except (OSError, json.JSONDecodeError):
            attempt_token_usages.append({})

        verdicts = _read_verdicts(run_outcome.normalised_paths[combo_key])
        errored_uids: List[str] = []
        for v in verdicts:
            all_verdicts[v.uid] = v
            if v.verdict_status == STATUS_ERROR:
                errored_uids.append(v.uid)

        if not errored_uids:
            break
        if attempt == cfg.max_retries:
            for uid in errored_uids:
                retry_events.append({
                    "uid": uid,
                    "attempt": attempt + 1,
                    "resolved": False,
                    "final_status": STATUS_ERROR,
                })
            break

        # Exponential backoff with mild jitter.
        delay = min(
            cfg.retry_max_delay,
            cfg.retry_base_delay * (2 ** attempt),
        )
        jitter = random.uniform(0.0, cfg.retry_base_delay * 0.25)
        sleep_seconds = round(delay + jitter, 3)
        for uid in errored_uids:
            retry_events.append({
                "uid": uid,
                "attempt": attempt + 1,
                "sleep_seconds": sleep_seconds,
                "next_attempt": attempt + 2,
            })
        sleep_fn(sleep_seconds)
        errored_set = set(errored_uids)
        pending_records = [r for r in pending_records if r.get("uid") in errored_set]

    # Defense-in-depth: both real adapters guarantee one verdict per input
    # uid (missing-uid fallback synthesises STATUS_ERROR). If a future
    # adapter regresses or a runner crash swallows a uid, we synthesise
    # STATUS_ERROR here so the record still has a deterministic fate rather
    # than silently vanishing from the cascade.
    for record in records:
        uid = record.get("uid")
        if uid and uid not in all_verdicts:
            all_verdicts[uid] = PoserVerdict(
                uid=uid,
                source=record.get("source", ""),
                verdict_status=STATUS_ERROR,
                verdict_score=0.0,
                poser_name=stage.combo.key(),
                poser_model="",
                verdict_detail={
                    "error_reason": "verdict missing from adapter output",
                    "provider": stage.combo.provider,
                },
            )

    # Merge phase: overwrite the stage top-level view with the last
    # verdict for every uid (input-order preserved), and derive the
    # survivor records list.
    stage_normalised_path = stage_subdir / f"{stage.combo.slug()}_normalised.jsonl"
    passing_uids: set = set()
    with stage_normalised_path.open("w", encoding="utf-8") as fh:
        for uid in [r.get("uid") for r in records]:
            v = all_verdicts.get(uid)
            if v is None:
                continue
            fh.write(json.dumps(v.to_jsonl_row()) + "\n")
            if v.verdict_status == STATUS_WELL_POSED:
                passing_uids.add(uid)

    stage_passed_path = stage_subdir / "passed_records.jsonl"
    with stage_passed_path.open("w", encoding="utf-8") as fh:
        for record in records:
            if record.get("uid") in passing_uids:
                fh.write(json.dumps(record) + "\n")

    counts: dict = {}
    for v in all_verdicts.values():
        counts[v.verdict_status] = counts.get(v.verdict_status, 0) + 1

    token_usage = _sum_token_usage(attempt_token_usages)
    stage_cost = _sum_estimated_cost(cfg, token_usage)

    return CascadeStageOutcome(
        stage=stage,
        wellposed_manifest_path=attempt_manifests[0] if attempt_manifests else stage_subdir / "run_manifest.json",
        normalised_path=stage_normalised_path,
        passed_records_path=stage_passed_path,
        input_uid_count=len(records),
        survivor_uid_count=len(passing_uids),
        counts=counts,
        wall_clock_seconds=0.0,
        token_usage=token_usage,
        estimated_cost_usd=stage_cost,
        retry_events=retry_events,
    )


def _read_verdicts(path: Path) -> List[PoserVerdict]:
    verdicts: List[PoserVerdict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            verdicts.append(PoserVerdict(
                uid=row["uid"],
                source=row.get("source", ""),
                verdict_status=row["verdict_status"],
                verdict_score=row.get("verdict_score", 0.0),
                poser_name=row.get("poser_name", ""),
                poser_model=row.get("poser_model", ""),
                verdict_detail=row.get("verdict_detail", {}) or {},
                verdict_signals=row.get("verdict_signals", {}) or {},
                raw_payload=row.get("raw_payload", {}) or {},
            ))
    return verdicts


def _load_records_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _sum_token_usage(usages: Iterable[dict]) -> dict:
    """Sum fleet_totals across per-attempt or per-stage token_usage blocks."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "records_with_usage": 0,
    }
    for u in usages:
        ft = (u or {}).get("fleet_totals") or {}
        for k in totals:
            totals[k] += int(ft.get(k) or 0)
    return {"fleet_totals": totals}


def _sum_estimated_cost(cfg: CascadeConfig, token_usage: dict) -> Optional[float]:
    if cfg.cost_per_input_mtok is None and cfg.cost_per_output_mtok is None:
        return None
    in_rate = cfg.cost_per_input_mtok or 0.0
    out_rate = cfg.cost_per_output_mtok or 0.0
    ft = token_usage.get("fleet_totals") or {}
    return round(
        ft.get("input_tokens", 0) / 1_000_000 * in_rate
        + ft.get("output_tokens", 0) / 1_000_000 * out_rate,
        6,
    )


def _stage_manifest_entry(so: CascadeStageOutcome) -> dict:
    return {
        "index": so.stage.index,
        "combo": so.stage.combo.key(),
        "slug": so.stage.combo.slug(),
        "input_uid_count": so.input_uid_count,
        "survivor_uid_count": so.survivor_uid_count,
        "counts": so.counts,
        "wall_clock_seconds": so.wall_clock_seconds,
        "wellposed_manifest_path": str(so.wellposed_manifest_path),
        "normalised_path": str(so.normalised_path),
        "passed_records_path": str(so.passed_records_path),
        "token_usage": so.token_usage,
        "estimated_cost_usd": so.estimated_cost_usd,
        "retry_events": so.retry_events,
    }
