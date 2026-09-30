"""Content-addressed run records. No credentials or unrestricted environment dump."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
import os
from pathlib import Path
import platform
import shutil
import sys

from .config import ReasoningConfig


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(observation, output_dir: Path, *, mode: str, result):
    package = Path(__file__).resolve().parents[1]
    code = {p.relative_to(package).as_posix(): file_hash(p) for p in sorted(package.rglob("*.py"))}
    fingerprint = hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest()
    archive = output_dir / "code_snapshot" / fingerprint
    if not archive.exists():
        shutil.copytree(package, archive, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    packages = {}
    for name in ["numpy", "scipy", "astropy", "photutils", "sep", "pydantic", "matplotlib", "Pillow"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    inputs = {}
    for label, value in [("image", observation.input_path), ("vision_arrays", observation.artifacts.get("vision_arrays"))]:
        if value:
            path = Path(value)
            inputs[label] = {"path": str(path), "sha256": file_hash(path) if path.is_file() else None,
                             "available": path.is_file()}
    manifest = {
        "manifest_version": 1, "mode": mode,
        "inputs": inputs, "code_sha256": fingerprint, "source_files": code,
        "code_snapshot": str(archive.resolve()),
        "python": sys.version, "platform": platform.platform(), "packages": packages,
        "numeric_thread_environment": {name: os.environ.get(name) for name in
            ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
        "vision_parameters": observation.provenance.parameters,
        "reasoning_parameters": asdict(ReasoningConfig()),
        "source_observation_sha256": result.evidence_preparation.get("source_observation_sha256"),
        "published_grade": result.final_decision["published_grade"],
        "audit_passed": result.decision_audit.audit_passed,
        "limitations": ["Hash plus snapshot enables attribution, not cross-platform bitwise guarantees.",
                        "Same-code replay verifies implementation consistency, not scientific accuracy.",
                        "Input image hash in reason mode is context only: that mode consumes Observation JSON."],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
