"""Bake model weights into an image layer and fingerprint them (W5.3).

Runtime is offline-first (``MODEL_WEIGHTS_OFFLINE=1`` / ``HF_HUB_OFFLINE=1``): a
deployment that needs the reranker or the dense embedding model must download it
here, at build time, instead of on a user's first request.

This is the one place in the project where a download is intended, so
``local_files_only=False`` is passed explicitly. Failures are loud: a build that
silently produced an image without weights would serve lexical-only reranking
while looking healthy.

The fingerprint (per-file sha256 of the downloaded snapshots) is what the SBOM /
release evidence can pin, so "which weights are in this image" is answerable
without trusting a tag.

Usage::

    python -m medcopilot.scripts.prepare_weights --cache-dir /opt/hf-cache
    python -m medcopilot.scripts.prepare_weights --skip-embedding   # reranker only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FINGERPRINT_SCHEMA_VERSION = 1
DEFAULT_CACHE_DIR = "/opt/hf-cache"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _snapshot_files(model_name: str, cache_dir: str) -> list[Path]:
    """Files of the snapshot HF resolved for ``model_name`` (empty when absent)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return []
    try:
        snapshot = Path(snapshot_download(model_name, cache_dir=cache_dir, local_files_only=True))
    except Exception:
        return []
    return sorted(path for path in snapshot.rglob("*") if path.is_file())


def download_model(kind: str, model_name: str, cache_dir: str) -> dict:
    """Fetch one model and return its fingerprint record; raises on failure."""
    if kind == "reranker":
        from sentence_transformers import CrossEncoder

        # local_files_only=False on purpose: baking may download (see docstring).
        CrossEncoder(model_name, cache_dir=cache_dir, local_files_only=False)
    else:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(model_name, cache_folder=cache_dir, local_files_only=False)

    files = _snapshot_files(model_name, cache_dir)
    if not files:
        raise RuntimeError(
            f"{kind} weights for '{model_name}' are not present in {cache_dir} after download"
        )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return {
        "kind": kind,
        "model_name": model_name,
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "sha256": digest.hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="bake model weights and write their fingerprint")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--fingerprint", default=None, help="default <cache-dir>/weights_fingerprint.json")
    parser.add_argument("--embedding", default=None, help="default EMBEDDING_MODEL")
    parser.add_argument("--reranker", default=None, help="default RERANKER_MODEL")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    args = parser.parse_args(argv)

    from medcopilot.common.config import config

    cache_dir = args.cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    targets: list[tuple[str, str]] = []
    if not args.skip_reranker:
        targets.append(("reranker", args.reranker or config.reranker_model))
    if not args.skip_embedding:
        targets.append(("embedding", args.embedding or config.embedding_model))

    records: list[dict] = []
    for kind, model_name in targets:
        try:
            record = download_model(kind, model_name, cache_dir)
        except Exception as exc:
            print(f"FAIL: could not bake {kind} weights '{model_name}': {exc}", file=sys.stderr)
            return 1
        records.append(record)
        print(f"baked {kind}: {model_name} ({record['file_count']} files, {record['total_bytes']} bytes)")

    fingerprint = {
        "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cache_dir": cache_dir,
        "models": records,
    }
    target = Path(args.fingerprint or (Path(cache_dir) / "weights_fingerprint.json"))
    target.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WEIGHTS FINGERPRINT: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
