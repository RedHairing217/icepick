#!/usr/bin/env python3
# Adaptive fold of batch 14 band records into corpus_pde625.
# Designed to fire UNATTENDED from gate_batch14_fold.sh when pass@k completes,
# while parallel sessions may be mutating the corpus concurrently. Therefore:
#   - reads CURRENT corpus state at run time (no hardcoded pre-fold count)
#   - idempotent: exits 0 if batch14 already folded
#   - best-effort mkdir lock to serialize against another copy of itself
#   - byte-identity within-batch dedup + vs-corpus collision guard
#   - backup + post-write all-uid-uniqueness verify (aborts before write on any violation)
import json, shutil, os, sys, time

CORPUS = "/Users/redhairing/Desktop/helloworld/icepick/out/corpus_pde625"
PASSK  = "/Users/redhairing/Desktop/helloworld/icepick/out/processing_20260709T062552Z/pass_at_k/pass_at_k.jsonl"
SRC_REL = "out/processing_20260709T062552Z/pass_at_k/pass_at_k.jsonl"
KEY = "batch14"
SOURCE_BATCH = "batch14_20260709T062552Z"
WELLPOSED_VIA = "cascade_sonnet_only_codex_anthropic"
BAK = ".bak-pre-batch14fold"
LOCK = f"{CORPUS}/.corpus_fold.lock"

def lj(p): return [json.loads(l) for l in open(p) if l.strip()]
def lJ(p): return json.load(open(p))
def log(m): print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {m}", flush=True)

# ---- best-effort lock (serialize against another instance of this script) ----
for _ in range(120):
    try:
        os.mkdir(LOCK); break
    except FileExistsError:
        time.sleep(5)
else:
    log("could not acquire corpus fold lock after 10min — aborting"); sys.exit(2)

try:
    if not os.path.exists(PASSK):
        log(f"pass@k output missing: {PASSK} — aborting"); sys.exit(1)

    manifest = lJ(f"{CORPUS}/corpus_manifest.json")
    if KEY in manifest["assembled_from"]:
        log(f"{KEY} already folded (manifest) — nothing to do, exiting idempotently"); sys.exit(0)

    wp_all = lJ(f"{CORPUS}/wellposed_all_with_passk.json")
    wp_band = lJ(f"{CORPUS}/wellposed_band.json")
    band_jsonl = lj(f"{CORPUS}/band_corpus.jsonl")
    pre_all, pre_band = len(wp_all), len(wp_band)
    log(f"live pre-fold state: wp_all={pre_all} wp_band={pre_band} band_jsonl={len(band_jsonl)} total_band={manifest['total_band_records']}")

    # batch14 pass@k, within-batch byte-identity dedup
    raw_rows = lj(PASSK)
    by_uid, dropped = {}, 0
    for r in raw_rows:
        u = r.get("uid")
        if not u:
            log("batch14 record with null uid — aborting"); sys.exit(1)
        if u in by_uid:
            if json.dumps(by_uid[u], sort_keys=True) != json.dumps(r, sort_keys=True):
                log(f"non-identical dup uid {u} in batch14 — aborting"); sys.exit(1)
            dropped += 1; continue
        by_uid[u] = r
    raw = list(by_uid.values())
    band = [r for r in raw if r.get("label") == "band"]
    log(f"batch14: {len(raw_rows)} rows -> {len(raw)} distinct WP (dropped {dropped}), {len(band)} band")
    if not band:
        log("batch14 has 0 band records — nothing to fold, exiting"); sys.exit(0)

    # collision guard vs live corpus
    corpus_uids = {r.get("uid") for r in wp_all if r.get("uid")}
    coll = {r["uid"] for r in raw} & corpus_uids
    if coll:
        log(f"uid collision with live corpus: {sorted(coll)[:5]} — aborting (batch14 should be fresh 2025-12)"); sys.exit(1)

    def enrich(r):
        return {
            "uid": r.get("uid"), "arxiv_id": r.get("arxiv_id"), "statement": r.get("statement"),
            "answer": r.get("answer"), "tier": r.get("tier"), "family": r.get("family"),
            "provenance": r.get("provenance"), "truth_policy": r.get("truth_policy"),
            "source": r.get("source"), "metadata": r.get("metadata"),
            "wellposed_via": WELLPOSED_VIA, "source_batch": SOURCE_BATCH,
            "pass_at_k_results": {
                "k": 8, "label": r.get("label"), "modal_wrong": r.get("modal_wrong"),
                "n_correct": r.get("n_correct"), "n_degenerate": r.get("n_degenerate"),
                "n_wrong": r.get("n_wrong"), "pass_at_k": r.get("pass_at_k"),
                "source_file": SRC_REL, "status": "final", "top_wrong_share": r.get("top_wrong_share"),
            },
        }
    def band_row(r):
        row = dict(r); row["corpus_provenance"] = {"batch": KEY, "source_file": SRC_REL}; return row

    new_wp_all = wp_all + [enrich(r) for r in raw]          # all WP (band + non-band) into wellposed_all
    new_wp_band = wp_band + [enrich(r) for r in band]
    new_band_jsonl = band_jsonl + [band_row(r) for r in band]
    manifest["assembled_from"][KEY] = {"file": SRC_REL, "band_count": len(band),
        "note": f"realmath math.AP 2025-12, Sonnet-only single-stage codex:anthropic cascade (226->{len(raw)} WP). "
                f"Folded unattended via gate on pass@k completion. {dropped} byte-identical dup(s) collapsed."}
    manifest["total_band_records"] = len(new_wp_band)

    # hard invariants (abort before writing on any violation)
    au = [r.get("uid") for r in new_wp_all if r.get("uid")]
    assert len(au) == len(new_wp_all) == len(set(au)), "post-fold wp_all uids not all-unique — ABORT, no write"
    bju = [r.get("uid") for r in new_band_jsonl if r.get("uid")]
    assert len(bju) == len(new_band_jsonl) == len(set(bju)), "post-fold band_jsonl uids not all-unique — ABORT, no write"
    assert len(new_wp_band) == len(new_band_jsonl) == manifest["total_band_records"], "band count mismatch — ABORT"

    for f in ["wellposed_all_with_passk.json", "wellposed_band.json", "band_corpus.jsonl", "corpus_manifest.json"]:
        src = f"{CORPUS}/{f}"
        if not os.path.exists(src + BAK):
            shutil.copy2(src, src + BAK)
    log(f"backups written ({BAK})")

    with open(f"{CORPUS}/wellposed_all_with_passk.json", "w") as f: json.dump(new_wp_all, f, ensure_ascii=False, indent=1)
    with open(f"{CORPUS}/wellposed_band.json", "w") as f: json.dump(new_wp_band, f, ensure_ascii=False, indent=1)
    with open(f"{CORPUS}/band_corpus.jsonl", "w") as f:
        for r in new_band_jsonl: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{CORPUS}/corpus_manifest.json", "w") as f: json.dump(manifest, f, ensure_ascii=False, indent=1)
    log(f"FOLDED batch14: +{len(raw)} WP / +{len(band)} band -> corpus now {len(new_wp_all)} WP / {len(new_wp_band)} band")
finally:
    try: os.rmdir(LOCK)
    except OSError: pass
