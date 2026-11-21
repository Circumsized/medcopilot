"""Versioned evaluation harnesses for Med-Copilot (offline, deterministic first)."""
from medcopilot.evaluation.golden_eval import (
    GOLDEN_DATASET_VERSION,
    CaseResult,
    GoldenDataset,
    GoldenDatasetError,
    GoldenReport,
    load_golden_dataset,
    run_golden,
    write_failure_samples,
)

__all__ = [
    "GOLDEN_DATASET_VERSION",
    "CaseResult",
    "GoldenDataset",
    "GoldenDatasetError",
    "GoldenReport",
    "load_golden_dataset",
    "run_golden",
    "write_failure_samples",
]
