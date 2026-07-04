"""Pipeline orchestrator — chains groundtruth → poser (± pass@k) into one command.

Two order shapes are supported via ``order``:

  ``"classic"`` (default) — the well-posedness gate runs before pass@k:
    input → groundtruth → wellposed → pass_at_k? → final

  ``"solvable-first"`` — pass@k runs first so the expensive wellposed
    cascade only scores solvable records:
    input → groundtruth → pass_at_k → filter drop → wellposed → final

Pass@k is optional. When ``pass_at_k_cfg is None`` the pipeline collapses
back to the historical ``groundtruth → wellposed`` chain, unchanged.

Each stage is run with its own config; the runner threads the previous
stage's record-output file as the next stage's input. The final corpus
is a stable top-level ``final_corpus.jsonl`` pointing at whichever
stage's output represents the fully-filtered set.

This is a convenience wrapper around the stage runners — it doesn't
re-implement them. Operators can still run the stages individually if
they want intermediate inspection or a different order.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from icepick.processing.groundtruth.config import GroundtruthConfig
from icepick.processing.groundtruth.runner import run as run_groundtruth
from icepick.processing.pass_at_k.config import PassAtKConfig
from icepick.processing.pass_at_k.runner import run as run_pass_at_k
from icepick.processing.poser.config import WellposedConfig
from icepick.processing.poser.runner import run as run_wellposed

ORDER_CLASSIC = "classic"
ORDER_SOLVABLE_FIRST = "solvable-first"
ORDER_VALUES = (ORDER_CLASSIC, ORDER_SOLVABLE_FIRST)

# Pass@k labels the pipeline treats as "not worth continuing to the next
# stage" in solvable-first order. Anything below the band is filtered so
# the expensive wellposed cascade only scores records the subject model
# can actually attempt (``band`` and ``solved``). Well-posedness is a
# separate check from solvability — filtering unsolvable records here is
# a cost lever, not a correctness one.
_UNSOLVABLE_LABELS = frozenset({"drop", "collapse", "misdirection"})


@dataclass
class PipelineOutcome:
    """Returned by ``run``. Paths are concrete; counts come from each stage."""

    final_corpus_path: Path
    final_corpus_count: int
    groundtruth_manifest_path: Path
    poser_manifest_path: Path
    groundtruth_counts: dict
    poser_counts: dict
    manifest_path: Path
    pass_at_k_manifest_path: Optional[Path] = None
    pass_at_k_counts: dict = field(default_factory=dict)
    order: str = ORDER_CLASSIC


def run(
    *,
    input_path: Path,
    output_dir: Path,
    groundtruth_cfg: GroundtruthConfig,
    poser_cfg: WellposedConfig,
    pass_at_k_cfg: Optional[PassAtKConfig] = None,
    order: str = ORDER_CLASSIC,
    groundtruth_adapter=None,
    poser_adapter_overrides=None,
    pass_at_k_backend=None,
) -> PipelineOutcome:
    """Run the pipeline stages in the requested order and write a final corpus.

    ``pass_at_k_cfg`` is optional — when unset the pipeline collapses to
    the historical ``groundtruth → wellposed`` chain. ``order`` selects
    between ``classic`` (wellposed then pass@k) and ``solvable-first``
    (pass@k then wellposed; unsolvable records are filtered before the
    expensive well-posedness gate runs).

    ``groundtruth_adapter``, ``poser_adapter_overrides``, and
    ``pass_at_k_backend`` are injection points for tests; production
    callers leave them None.
    """
    if order not in ORDER_VALUES:
        raise ValueError(f"order must be one of {ORDER_VALUES}, got {order!r}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: groundtruth. Force its output_dir under our umbrella.
    groundtruth_cfg.output_dir = output_dir / "groundtruth"
    gt_records = list(_iter_jsonl(input_path))
    gt_outcome = run_groundtruth(
        cfg=groundtruth_cfg,
        records=gt_records,
        adapter=groundtruth_adapter,
    )

    published_records = list(_iter_jsonl(gt_outcome.published_path))
    if not published_records:
        return _write_empty_pipeline(
            input_path=input_path,
            output_dir=output_dir,
            groundtruth_outcome=gt_outcome,
            order=order,
        )

    # Wire the intermediate stages by order. Both orders finish with the
    # last stage's records as the final corpus.
    poser_cfg.output_dir = output_dir / "wellposed"
    if pass_at_k_cfg is not None:
        pass_at_k_cfg.output_dir = output_dir / "pass_at_k"

    if pass_at_k_cfg is None:
        poser_outcome = run_wellposed(
            cfg=poser_cfg, records=published_records,
            adapter_overrides=poser_adapter_overrides,
        )
        return _finalise(
            input_path=input_path, output_dir=output_dir,
            groundtruth_outcome=gt_outcome, poser_outcome=poser_outcome,
            pass_at_k_outcome=None, order=order,
            final_source=poser_outcome.passed_records_path,
        )

    if order == ORDER_CLASSIC:
        # wellposed → pass@k. Pass@k scores well-posed survivors, so
        # only proven-worthy records incur k rollouts each.
        poser_outcome = run_wellposed(
            cfg=poser_cfg, records=published_records,
            adapter_overrides=poser_adapter_overrides,
        )
        pak_input = list(_iter_jsonl(poser_outcome.passed_records_path))
        if not pak_input:
            return _finalise(
                input_path=input_path, output_dir=output_dir,
                groundtruth_outcome=gt_outcome, poser_outcome=poser_outcome,
                pass_at_k_outcome=None, order=order,
                final_source=poser_outcome.passed_records_path,
            )
        pak_outcome = run_pass_at_k(
            cfg=pass_at_k_cfg, records=pak_input, backend=pass_at_k_backend,
        )
        return _finalise(
            input_path=input_path, output_dir=output_dir,
            groundtruth_outcome=gt_outcome, poser_outcome=poser_outcome,
            pass_at_k_outcome=pak_outcome, order=order,
            final_source=pak_outcome.output_path,
        )

    # ORDER_SOLVABLE_FIRST: pass@k → filter → wellposed. Pass@k stamps
    # every record; ``label`` values in _UNSOLVABLE_LABELS are dropped so
    # the wellposed cascade skips records no model could score.
    pak_outcome = run_pass_at_k(
        cfg=pass_at_k_cfg, records=published_records, backend=pass_at_k_backend,
    )
    solvable_records = _filter_solvable(pak_outcome.output_path)
    if not solvable_records:
        # Pass@k dropped everything — nothing worth well-posedness testing.
        return _finalise(
            input_path=input_path, output_dir=output_dir,
            groundtruth_outcome=gt_outcome, poser_outcome=None,
            pass_at_k_outcome=pak_outcome, order=order,
            final_source=None,
        )
    poser_outcome = run_wellposed(
        cfg=poser_cfg, records=solvable_records,
        adapter_overrides=poser_adapter_overrides,
    )
    return _finalise(
        input_path=input_path, output_dir=output_dir,
        groundtruth_outcome=gt_outcome, poser_outcome=poser_outcome,
        pass_at_k_outcome=pak_outcome, order=order,
        final_source=poser_outcome.passed_records_path,
    )


def _filter_solvable(pak_output_path: Path) -> list:
    """Load pass@k output rows, drop label∈_UNSOLVABLE_LABELS."""
    rows = list(_iter_jsonl(pak_output_path))
    return [r for r in rows if r.get("label") not in _UNSOLVABLE_LABELS]


def _finalise(
    *,
    input_path,
    output_dir: Path,
    groundtruth_outcome,
    poser_outcome,
    pass_at_k_outcome,
    order: str,
    final_source: Optional[Path],
) -> PipelineOutcome:
    """Materialise the top-level final_corpus.jsonl + pipeline manifest."""
    final_corpus_path = output_dir / "final_corpus.jsonl"
    if final_source is None:
        final_corpus_path.write_text("")
        final_count = 0
    else:
        shutil.copyfile(final_source, final_corpus_path)
        final_count = _count_lines(final_corpus_path)

    manifest = _build_pipeline_manifest(
        input_path=input_path,
        output_dir=output_dir,
        groundtruth_outcome=groundtruth_outcome,
        poser_outcome=poser_outcome,
        pass_at_k_outcome=pass_at_k_outcome,
        order=order,
        final_corpus_path=final_corpus_path,
        final_corpus_count=final_count,
    )
    manifest_path = output_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return PipelineOutcome(
        final_corpus_path=final_corpus_path,
        final_corpus_count=final_count,
        groundtruth_manifest_path=groundtruth_outcome.manifest_path,
        poser_manifest_path=(
            poser_outcome.manifest_path if poser_outcome else output_dir / "wellposed" / "run_manifest.json"
        ),
        groundtruth_counts=groundtruth_outcome.counts,
        poser_counts=poser_outcome.counts if poser_outcome else {},
        pass_at_k_manifest_path=(pass_at_k_outcome.manifest_path if pass_at_k_outcome else None),
        pass_at_k_counts=(pass_at_k_outcome.counts if pass_at_k_outcome else {}),
        order=order,
        manifest_path=manifest_path,
    )


def _write_empty_pipeline(
    *, input_path, output_dir: Path, groundtruth_outcome, order: str,
) -> PipelineOutcome:
    """Groundtruth admitted nothing — skip everything downstream."""
    final_corpus_path = output_dir / "final_corpus.jsonl"
    final_corpus_path.write_text("")
    manifest = _build_pipeline_manifest(
        input_path=input_path, output_dir=output_dir,
        groundtruth_outcome=groundtruth_outcome, poser_outcome=None,
        pass_at_k_outcome=None, order=order,
        final_corpus_path=final_corpus_path, final_corpus_count=0,
    )
    manifest_path = output_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return PipelineOutcome(
        final_corpus_path=final_corpus_path,
        final_corpus_count=0,
        groundtruth_manifest_path=groundtruth_outcome.manifest_path,
        poser_manifest_path=output_dir / "wellposed" / "run_manifest.json",
        groundtruth_counts=groundtruth_outcome.counts,
        poser_counts={},
        order=order,
        manifest_path=manifest_path,
    )


def _iter_jsonl(path) -> Iterable[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"input not found: {path}")
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def _build_pipeline_manifest(
    *,
    input_path,
    output_dir,
    groundtruth_outcome,
    poser_outcome,
    pass_at_k_outcome=None,
    order: str = ORDER_CLASSIC,
    final_corpus_path,
    final_corpus_count,
) -> dict:
    """Top-level manifest pointing at every stage manifest below.

    Stage order in the manifest mirrors execution order — classic runs
    wellposed then pass@k; solvable-first runs pass@k then wellposed.
    """
    stages = [
        {
            "stage": "groundtruth",
            "manifest_path": str(groundtruth_outcome.manifest_path),
            "counts": groundtruth_outcome.counts,
        }
    ]
    downstream = []
    if order == ORDER_CLASSIC:
        if poser_outcome is not None:
            downstream.append(("wellposed", poser_outcome))
        if pass_at_k_outcome is not None:
            downstream.append(("pass_at_k", pass_at_k_outcome))
    else:
        if pass_at_k_outcome is not None:
            downstream.append(("pass_at_k", pass_at_k_outcome))
        if poser_outcome is not None:
            downstream.append(("wellposed", poser_outcome))
    for name, outcome in downstream:
        stages.append({
            "stage": name,
            "manifest_path": str(outcome.manifest_path),
            "counts": outcome.counts,
        })
    return {
        "stage": "pipeline",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "order": order,
        "stages": stages,
        "final_corpus": {
            "path": str(final_corpus_path),
            "record_count": final_corpus_count,
        },
    }
