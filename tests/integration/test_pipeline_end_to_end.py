"""End-to-end pipeline test: ingest → groundtruth → poser → final corpus.

Uses fake adapters for both stages so no Anthropic / poser subprocess
calls happen. Verifies:

  - records flow through both stages in order
  - groundtruth's published.jsonl becomes the poser's input
  - the poser's passed_records.jsonl becomes the final corpus
  - generated records are dropped at groundtruth, never reach the poser
  - the pipeline manifest points at every stage manifest
  - the solvable-first order threads pass@k before wellposed and drops
    unsolvable records before the expensive well-posedness cascade runs
"""

from __future__ import annotations

import json
from pathlib import Path

from icepick.processing.groundtruth.base import (
    STATUS_PUBLISHED,
    STATUS_UNPUBLISHED,
    GroundtruthVerdict,
)
from icepick.processing.groundtruth.config import GroundtruthConfig
from icepick.processing.pass_at_k.config import PassAtKConfig
from icepick.processing.pipeline import run as run_pipeline
from icepick.processing.poser.base import (
    STATUS_ILL_POSED,
    STATUS_WELL_POSED,
    PoserRequest,
    PoserRunResult,
    PoserVerdict,
)
from icepick.processing.poser.config import (
    BUILD_CLAUDE,
    PROVIDER_ANTHROPIC,
    Combo,
    WellposedConfig,
)


class _FakePassAtKBackend:
    """Returns k rollouts where correctness is baked into the question text.

    A statement containing ``"solvable"`` answers correctly on every roll;
    a statement containing ``"unsolvable"`` answers wrong every time and
    hits label=drop territory (pass@k=0).
    """

    name = "fake"

    def __init__(self):
        self._input_tokens = 0
        self._output_tokens = 0

    def call(self, question, *, k, temperature, max_tokens, think, timeout):
        # The subject-model output is boxed answer text — the scoring
        # layer extracts \\boxed{...}. "solvable" records answer "1";
        # "unsolvable" records answer "0" (verifier sees ≠ truth).
        if "solvable" in question and "unsolvable" not in question:
            return [r"\boxed{1}"] * k
        return [r"\boxed{0}"] * k

    def usage(self):
        return {"input_tokens": 0, "output_tokens": 0}


class _FakeGroundtruthAdapter:
    """Pass for papers ending in EVEN digit, fail for odd."""

    def lookup_paper(self, *, arxiv_id, paper_title, uid_for_error_attribution):
        last = arxiv_id[-1]
        status = STATUS_PUBLISHED if last in "02468" else STATUS_UNPUBLISHED
        return GroundtruthVerdict(
            uid=uid_for_error_attribution, source="",
            verdict_status=status, arxiv_id=arxiv_id,
            judge_model="fake-gt", judge_votes=[status] * 3,
            judge_majority=status, reasoning="fake", confidence="high",
        )


class _FakePoserAdapter:
    """Pass any record whose statement contains 'good', fail otherwise."""

    build = "claude"

    def plan(self, records, cfg, combo, work_dir):
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        # Mirror what the real adapters do: write the uid-injected input
        # to disk so the runner has a single canonical record source.
        input_path = Path(work_dir) / f"{combo.slug()}_input.jsonl"
        with input_path.open("w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        return PoserRequest(
            argv=["fake-poser"], env={},
            input_path=input_path,
            output_path=Path(work_dir) / f"{combo.slug()}_out.json",
            cache_path=None, poser_name=combo.key(),
        )

    def run(self, request):
        return PoserRunResult(
            exit_code=0, stdout="", stderr="",
            output_path=request.output_path, wall_clock_seconds=0.01,
        )

    def normalise(self, raw_output_path, input_uids, *, combo):
        # We need to know which uids had "good" in the statement, but the
        # adapter only sees uids here. Look it up from the input file the
        # plan() phase wrote.
        input_path = raw_output_path.parent / f"{combo.slug()}_input.jsonl"
        if not input_path.exists():
            # Fallback: emit defer for every uid
            return [PoserVerdict(uid=u, source="", verdict_status="defer",
                                 verdict_score=0.5, poser_name=combo.key(),
                                 poser_model="fake") for u in input_uids]
        good_uids = set()
        with input_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "good" in (row.get("statement") or "").lower():
                    good_uids.add(row.get("uid"))
        return [
            PoserVerdict(
                uid=u, source="",
                verdict_status=STATUS_WELL_POSED if u in good_uids else STATUS_ILL_POSED,
                verdict_score=1.0 if u in good_uids else 0.0,
                poser_name=combo.key(), poser_model="fake",
            )
            for u in input_uids
        ]


def _write_input(tmp_path: Path, records: list) -> Path:
    path = tmp_path / "raw_input.jsonl"
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return path


def test_pipeline_end_to_end_filters_through_both_stages(tmp_path):
    # 4 records:
    #   uid_pp: arxiv 2403.12340 (published) + "good" statement → final corpus
    #   uid_pf: arxiv 2403.12342 (published) + "bad" statement → fails poser
    #   uid_fp: arxiv 2403.12341 (unpublished) + "good" statement → fails groundtruth
    #   uid_gen: provenance=computed → discarded at groundtruth
    records = [
        {"source": "rm", "statement": "good theorem 1", "arxiv_id": "2403.12340",
         "provenance": "extracted", "uid": "uid_pp"},
        {"source": "rm", "statement": "bad theorem 2", "arxiv_id": "2403.12342",
         "provenance": "extracted", "uid": "uid_pf"},
        {"source": "rm", "statement": "good theorem 3", "arxiv_id": "2403.12341",
         "provenance": "extracted", "uid": "uid_fp"},
        {"source": "gen", "statement": "good but generated", "family": "calc",
         "provenance": "computed", "uid": "uid_gen"},
    ]
    input_path = _write_input(tmp_path, records)

    gt_cfg = GroundtruthConfig(
        mode="flow_testing",
        output_dir=tmp_path / "ignored_gt",  # pipeline overrides this
        calibration_sheet=tmp_path / "sheet.jsonl",
    )
    poser_cfg = WellposedConfig(
        combos=[Combo(build=BUILD_CLAUDE, provider=PROVIDER_ANTHROPIC)],
        mode="production",
        output_dir=tmp_path / "ignored_poser",  # pipeline overrides
        enable_judge_tier=False,
    )

    outcome = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "pipeline_out",
        groundtruth_cfg=gt_cfg,
        poser_cfg=poser_cfg,
        groundtruth_adapter=_FakeGroundtruthAdapter(),
        poser_adapter_overrides={BUILD_CLAUDE: _FakePoserAdapter()},
    )

    # --- final corpus contents
    final = [json.loads(l) for l in outcome.final_corpus_path.read_text().splitlines() if l.strip()]
    final_uids = {r["uid"] for r in final}
    assert final_uids == {"uid_pp"}, f"expected only uid_pp, got {final_uids}"
    assert outcome.final_corpus_count == 1

    # --- groundtruth stage filtered correctly
    gt_published = [json.loads(l) for l in (outcome.groundtruth_manifest_path.parent / "published.jsonl").read_text().splitlines() if l.strip()]
    assert {r["uid"] for r in gt_published} == {"uid_pp", "uid_pf"}, \
        "groundtruth should publish the two extracted+published records, drop unpublished + generated"

    # --- poser stage filtered correctly
    assert outcome.poser_counts.get(STATUS_WELL_POSED, 0) == 1
    assert outcome.poser_counts.get(STATUS_ILL_POSED, 0) == 1

    # --- top-level manifest references both stage manifests
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["stage"] == "pipeline"
    assert manifest["final_corpus"]["record_count"] == 1
    stages = {s["stage"] for s in manifest["stages"]}
    assert stages == {"groundtruth", "wellposed"}


def test_pipeline_handles_empty_groundtruth_output(tmp_path):
    """If groundtruth rejects everything, pipeline still completes cleanly."""
    records = [
        {"source": "rm", "statement": "x", "arxiv_id": "2403.12341",
         "provenance": "extracted", "uid": "uid_only"},
    ]
    input_path = _write_input(tmp_path, records)
    gt_cfg = GroundtruthConfig(
        mode="flow_testing",
        output_dir=tmp_path / "ignored",
        calibration_sheet=tmp_path / "sheet.jsonl",
    )
    poser_cfg = WellposedConfig(
        combos=[Combo(build=BUILD_CLAUDE, provider=PROVIDER_ANTHROPIC)],
        mode="production",
        output_dir=tmp_path / "ignored",
        enable_judge_tier=False,
    )

    outcome = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "out",
        groundtruth_cfg=gt_cfg,
        poser_cfg=poser_cfg,
        groundtruth_adapter=_FakeGroundtruthAdapter(),  # fails on arxiv ending in 1
        poser_adapter_overrides={BUILD_CLAUDE: _FakePoserAdapter()},
    )

    assert outcome.final_corpus_count == 0
    assert outcome.final_corpus_path.read_text() == ""
    # Poser never ran on records → poser_counts is empty
    assert outcome.poser_counts == {}


def test_pipeline_solvable_first_drops_unsolvable_before_wellposed(tmp_path):
    """Solvable-first order runs pass@k first; label=drop records get
    filtered out so the wellposed cascade never sees them."""
    # 4 records — all extracted, all groundtruth-published (arxiv ends in
    # even digit). Truth is "1" everywhere; the fake backend answers "1"
    # only when the statement contains "solvable".
    records = [
        {"source": "rm", "statement": "solvable good problem 1", "arxiv_id": "2403.12340",
         "provenance": "extracted", "answer": "1", "uid": "uid_solve_wp"},
        {"source": "rm", "statement": "solvable bad problem 2", "arxiv_id": "2403.12342",
         "provenance": "extracted", "answer": "1", "uid": "uid_solve_ip"},
        {"source": "rm", "statement": "unsolvable good problem 3", "arxiv_id": "2403.12344",
         "provenance": "extracted", "answer": "1", "uid": "uid_unsolve"},
        {"source": "rm", "statement": "unsolvable bad problem 4", "arxiv_id": "2403.12346",
         "provenance": "extracted", "answer": "1", "uid": "uid_unsolve2"},
    ]
    input_path = _write_input(tmp_path, records)

    gt_cfg = GroundtruthConfig(
        mode="flow_testing",
        output_dir=tmp_path / "ignored_gt",
        calibration_sheet=tmp_path / "sheet.jsonl",
    )
    poser_cfg = WellposedConfig(
        combos=[Combo(build=BUILD_CLAUDE, provider=PROVIDER_ANTHROPIC)],
        mode="production",
        output_dir=tmp_path / "ignored_poser",
        enable_judge_tier=False,
    )
    pak_cfg = PassAtKConfig(
        mode="production",
        output_dir=tmp_path / "ignored_pak",
        backend="qwen_http",  # local backend: no allow_live_calls required
        backend_url="http://localhost:0/v1/chat/completions",  # never dialled — fake backend
        model="fake-model",
        k=4,
        temperature=0.0,
        max_tokens=64,
        max_concurrent=1,  # deterministic ordering for the assertion
        max_retries=0,
    )

    outcome = run_pipeline(
        input_path=input_path,
        output_dir=tmp_path / "pipeline_out",
        groundtruth_cfg=gt_cfg,
        poser_cfg=poser_cfg,
        pass_at_k_cfg=pak_cfg,
        order="solvable-first",
        groundtruth_adapter=_FakeGroundtruthAdapter(),
        poser_adapter_overrides={BUILD_CLAUDE: _FakePoserAdapter()},
        pass_at_k_backend=_FakePassAtKBackend(),
    )

    # --- final corpus: only records that were solvable AND well-posed
    final = [json.loads(l) for l in outcome.final_corpus_path.read_text().splitlines() if l.strip()]
    assert {r["uid"] for r in final} == {"uid_solve_wp"}
    assert outcome.final_corpus_count == 1

    # --- pass@k saw all 4 groundtruth survivors (arxiv ends in even digit)
    assert outcome.pass_at_k_counts, "pass@k stage must have run in solvable-first mode"
    # solvable → label=solved (pass@k=1.0). unsolvable → pass@k=0 with a
    # single wrong-answer attractor, which the scorer labels below-band
    # (misdirection or collapse depending on the wrong-share threshold —
    # either is in _UNSOLVABLE_LABELS and gets filtered before wellposed).
    assert outcome.pass_at_k_counts.get("solved", 0) == 2
    filtered = (
        outcome.pass_at_k_counts.get("drop", 0)
        + outcome.pass_at_k_counts.get("collapse", 0)
        + outcome.pass_at_k_counts.get("misdirection", 0)
    )
    assert filtered == 2, f"expected 2 filtered-out records, got counts {outcome.pass_at_k_counts}"

    # --- wellposed only ran on the 2 solvable records
    assert outcome.poser_counts.get(STATUS_WELL_POSED, 0) == 1
    assert outcome.poser_counts.get(STATUS_ILL_POSED, 0) == 1
    assert sum(outcome.poser_counts.values()) == 2, \
        "wellposed must not have seen the 2 drop-labeled records"

    # --- manifest order reflects solvable-first
    manifest = json.loads(outcome.manifest_path.read_text())
    assert manifest["order"] == "solvable-first"
    stage_names = [s["stage"] for s in manifest["stages"]]
    assert stage_names == ["groundtruth", "pass_at_k", "wellposed"]
