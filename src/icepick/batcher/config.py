"""BatcherConfig — configuration dataclass for the bulk-batcher daemon.

IDENTITY-CRITICAL fields (campaign_source, slice_size) are persisted in
queue_state.json on first run. Subsequent runs refuse to start on mismatch:
changing campaign_source would fork uid identity, silently producing records
that the ledger cannot detect as duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_DEFAULT_KEY_PATH = "/Users/redhairing/Desktop/helloworld/anthro_key.env"
_DEFAULT_ROOT = "out/auto_batcher"


@dataclass
class BatcherConfig:
    # Required
    journal_path: Path
    run_dir: Path  # intake run dir owning the journal (for read_manifest_source_name/run_concluded)
    campaign_source: str

    # Optional with defaults
    root: Path = field(default_factory=lambda: Path(_DEFAULT_ROOT))
    cross_source_statement_policy: str = "skip"
    slice_size: int = 250
    cost_limit_usd: float = 5.0
    key_path: str = _DEFAULT_KEY_PATH
    mode: str = "production"
    calibration_sheet: Optional[str] = None
    icepick_bin: str = "icepick"
    poll_interval_s: int = 60
    qwen_recheck_interval_s: int = 45
    watch_journals: list = field(default_factory=list)  # [{label, journal_path, run_dir}]

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of all fields."""
        return {
            "root": str(self.root),
            "journal_path": str(self.journal_path),
            "run_dir": str(self.run_dir),
            "campaign_source": self.campaign_source,
            "cross_source_statement_policy": self.cross_source_statement_policy,
            "slice_size": self.slice_size,
            "cost_limit_usd": self.cost_limit_usd,
            "key_path": self.key_path,
            "mode": self.mode,
            "calibration_sheet": self.calibration_sheet,
            "icepick_bin": self.icepick_bin,
            "poll_interval_s": self.poll_interval_s,
            "qwen_recheck_interval_s": self.qwen_recheck_interval_s,
            "watch_journals": list(self.watch_journals),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BatcherConfig":
        """Reconstruct a BatcherConfig from a to_dict() output."""
        return cls(
            root=Path(d["root"]),
            journal_path=Path(d["journal_path"]),
            run_dir=Path(d["run_dir"]),
            campaign_source=d["campaign_source"],
            cross_source_statement_policy=d.get("cross_source_statement_policy", "skip"),
            slice_size=d.get("slice_size", 250),
            cost_limit_usd=d.get("cost_limit_usd", 5.0),
            key_path=d.get("key_path", _DEFAULT_KEY_PATH),
            mode=d.get("mode", "production"),
            calibration_sheet=d.get("calibration_sheet"),
            icepick_bin=d.get("icepick_bin", "icepick"),
            poll_interval_s=d.get("poll_interval_s", 60),
            qwen_recheck_interval_s=d.get("qwen_recheck_interval_s", 45),
            watch_journals=d.get("watch_journals", []),
        )

    # ------------------------------------------------------------------
    # Identity-critical field helpers
    # ------------------------------------------------------------------

    IDENTITY_FIELDS = ("campaign_source", "slice_size")

    def identity_dict(self) -> dict:
        """Return only the identity-critical fields."""
        return {k: getattr(self, k) for k in self.IDENTITY_FIELDS}

    def check_identity(self, persisted: dict) -> list[str]:
        """Return a list of mismatched field descriptions (empty = OK).

        Compares persisted identity fields against self.  Called on startup
        when queue_state.json already exists to refuse a mismatched campaign.
        """
        mismatches = []
        for k in self.IDENTITY_FIELDS:
            persisted_v = persisted.get(k)
            current_v = getattr(self, k)
            if persisted_v != current_v:
                mismatches.append(
                    f"{k}: persisted={persisted_v!r} vs current={current_v!r}"
                )
        return mismatches
