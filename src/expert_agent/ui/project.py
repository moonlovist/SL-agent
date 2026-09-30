from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable

import numpy as np


FITS_SUFFIXES = {".fits", ".fit", ".fts"}
GROUP_ORDER = ("A", "B", "C", "X", "无法判断", "运行中", "等待运行", "运行失败")
REFERENCE_GRADES = {"A", "B", "C", "X"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def object_id_from_input(path: str | Path) -> str:
    value = Path(path).expanduser().resolve()
    return value.stem if value.is_file() else value.name


def infer_reference_grade(path: str | Path) -> tuple[str | None, str | None]:
    """Read an explicit A/B/C/X label directory without guessing from IDs."""
    value = Path(path).expanduser().resolve()
    directories = value.parents if value.is_file() else (value, *value.parents)
    for directory in directories:
        label = directory.name.strip()
        match = re.fullmatch(
            r"([ABCX])(?:\s*样本|\s*[级类]|[-_ ]?(?:samples?|grade|class))?",
            label,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).upper(), f"导入目录：{directory}"
    return None, None


def discover_inputs(path: str | Path) -> list[Path]:
    """Find LensAgent inputs while preferring one object-named FITS per folder."""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        return [root] if root.suffix.lower() in FITS_SUFFIXES else []
    if not root.is_dir():
        return []
    preferred: list[Path] = []
    represented: set[Path] = set()
    for directory in [root, *sorted(item for item in root.rglob("*") if item.is_dir())]:
        candidate = directory / f"{directory.name}.fits"
        if candidate.is_file():
            preferred.append(candidate.resolve())
            represented.add(directory.resolve())
    extras = [
        item.resolve()
        for item in sorted(root.rglob("*"))
        if item.is_file()
        and item.suffix.lower() in FITS_SUFFIXES
        and item.parent.resolve() not in represented
    ]
    return list(dict.fromkeys([*preferred, *extras]))


def render_input_preview(
    input_path: str | Path,
    cache_dir: str | Path,
    *,
    band: str = "VIS",
) -> Path:
    """Write a display-only native FITS preview using the review UI stretch."""
    from PIL import Image

    from ..vision.preprocess import load_image

    image = load_image(input_path, band)
    values = np.asarray(image.data, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        shown = np.zeros(values.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(finite, [1.0, 99.7])
        span = max(float(high - low), np.finfo(float).eps)
        shown = np.asarray(
            np.clip((values - float(low)) / span, 0.0, 1.0) * 255.0,
            dtype=np.uint8,
        )
    shown = np.flipud(shown)
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{image.object_id}_{band}_linear.png"
    Image.fromarray(shown, mode="L").save(target)
    return target


@dataclass
class ResultSnapshot:
    object_id: str
    input_path: str
    output_dir: str
    grade: str = "无法判断"
    status: str = "completed"
    rationale: str = ""
    supporting: list[str] = field(default_factory=list)
    limiting: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    model_success: bool | None = None
    data_usable: bool | None = None
    colour_state: str = "unknown"
    audit_passed: bool | None = None
    needs_review: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def group(self) -> str:
        if self.status == "running":
            return "运行中"
        if self.status == "queued":
            return "等待运行"
        if self.status in {"failed", "cancelled", "interrupted"}:
            return "运行失败"
        return self.grade if self.grade in {"A", "B", "C", "X"} else "无法判断"


def _artifact_paths(values: dict[str, Any], output_dir: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, raw in values.items():
        if not isinstance(raw, str):
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = output_dir / candidate
        result[key] = str(candidate.resolve())
    return result


def load_result_snapshot(output_dir: str | Path) -> ResultSnapshot:
    output = Path(output_dir).expanduser().resolve()
    summary_path = output / "summary.json"
    observation_path = output / "observation.json"
    if not summary_path.exists() or not observation_path.exists():
        raise FileNotFoundError(f"Incomplete LensAgent result: {output}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    assessment = summary.get("expert_assessment") or {}
    dossier = summary.get("evidence_dossier") or {}
    audit = summary.get("decision_audit") or {}
    quality = observation.get("data_quality") or {}
    grade = summary.get("published_grade") or assessment.get("vote_category")
    if grade not in {"A", "B", "C", "X"}:
        grade = "无法判断"
    supporting = [
        str(item.get("statement", "")).strip()
        for item in dossier.get("supporting_evidence", [])
        if isinstance(item, dict) and str(item.get("statement", "")).strip()
    ]
    limiting = [
        str(item.get("statement") or item.get("description") or "").strip()
        for key in ("contradicting_evidence", "processing_cautions")
        for item in dossier.get(key, [])
        if isinstance(item, dict)
        and str(item.get("statement") or item.get("description") or "").strip()
    ]
    flags = [str(item) for item in quality.get("flags", [])]
    needs_review = bool(
        not quality.get("usable", False)
        or not quality.get("model_success", False)
        or "foreground_center_ambiguous" in flags
        or audit.get("abstention_recommended", False)
        or audit.get("required_corrections")
    )
    questions = assessment.get("assessment_questions") or {}
    return ResultSnapshot(
        object_id=str(observation.get("object_id") or summary.get("object_id") or output.name),
        input_path=str(observation.get("input_path") or ""),
        output_dir=str(output),
        grade=grade,
        rationale=str(assessment.get("assessment_rationale") or ""),
        supporting=supporting[:8],
        limiting=limiting[:8],
        artifacts=_artifact_paths(observation.get("artifacts") or {}, output),
        model_success=quality.get("model_success"),
        data_usable=quality.get("usable"),
        colour_state=str(questions.get("colour_evidence_state") or "unknown"),
        audit_passed=audit.get("audit_passed"),
        needs_review=needs_review,
        flags=flags,
    )


@dataclass
class RunRecord:
    run_id: str
    kind: str
    output_root: str
    output_dir: str
    status: str = "queued"
    started_utc: str | None = None
    finished_utc: str | None = None
    error: str | None = None
    grade: str | None = None
    parent_run_id: str | None = None


@dataclass
class ObjectRecord:
    object_id: str
    input_path: str
    reference_grade: str | None = None
    reference_grade_source: str | None = None
    reference_grade_confirmed: bool = False
    status: str = "queued"
    current_run_id: str | None = None
    runs: list[RunRecord] = field(default_factory=list)

    def current_run(self) -> RunRecord | None:
        return next(
            (item for item in reversed(self.runs) if item.run_id == self.current_run_id),
            self.runs[-1] if self.runs else None,
        )


class ProjectIndex:
    """Small persistent index; scientific JSON/FITS files remain authoritative."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "lensagent_ui_state.json"
        self.theme = "dark"
        self.objects: dict[str, ObjectRecord] = {}
        self.load()
        self.sync_completed_runs()

    def load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.theme = str(payload.get("theme", "dark"))
        for key, raw in (payload.get("objects") or {}).items():
            runs = [RunRecord(**item) for item in raw.get("runs", [])]
            status = str(raw.get("status", "queued"))
            if status in {"running", "reviewing"}:
                status = "interrupted"
            input_path = str(raw.get("input_path", ""))
            inferred_grade, inferred_source = infer_reference_grade(input_path)
            reference_grade = raw.get("reference_grade") or inferred_grade
            if reference_grade not in REFERENCE_GRADES:
                reference_grade = None
            self.objects[key] = ObjectRecord(
                object_id=str(raw.get("object_id", key)),
                input_path=input_path,
                reference_grade=reference_grade,
                reference_grade_source=(
                    raw.get("reference_grade_source") or inferred_source
                ),
                reference_grade_confirmed=bool(
                    raw.get("reference_grade_confirmed", False)
                    or reference_grade is not None
                ),
                status=status,
                current_run_id=raw.get("current_run_id"),
                runs=runs,
            )

    def save(self) -> None:
        payload = {
            "format_version": 1,
            "theme": self.theme,
            "objects": {key: asdict(value) for key, value in self.objects.items()},
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=self.root, suffix=".json"
        ) as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            temporary = Path(stream.name)
        temporary.replace(self.path)

    def sync_completed_runs(self) -> set[str]:
        """Refresh cached UI grades from authoritative completed-run JSON."""
        changed: set[str] = set()
        for object_id, record in self.objects.items():
            for run in record.runs:
                if run.status != "completed":
                    continue
                try:
                    snapshot = load_result_snapshot(run.output_dir)
                except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
                    continue
                if run.grade != snapshot.grade:
                    run.grade = snapshot.grade
                    changed.add(object_id)
        if changed:
            self.save()
        return changed

    def add_inputs(self, inputs: Iterable[str | Path]) -> list[ObjectRecord]:
        added: list[ObjectRecord] = []
        for raw in inputs:
            path = Path(raw).expanduser().resolve()
            object_id = object_id_from_input(path)
            existing = self.objects.get(object_id)
            if existing is not None and Path(existing.input_path) == path:
                continue
            reference_grade, reference_source = infer_reference_grade(path)
            record = ObjectRecord(
                object_id=object_id,
                input_path=str(path),
                reference_grade=reference_grade,
                reference_grade_source=reference_source,
                reference_grade_confirmed=reference_grade is not None,
            )
            self.objects[object_id] = record
            added.append(record)
        self.save()
        return added

    def new_run(
        self,
        record: ObjectRecord,
        *,
        kind: str,
        parent_run_id: str | None = None,
    ) -> RunRecord:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{kind}_{stamp}"
        output_root = self.root / "runs" / run_id
        run = RunRecord(
            run_id=run_id,
            kind=kind,
            output_root=str(output_root),
            output_dir=str(output_root / record.object_id),
            parent_run_id=parent_run_id,
        )
        record.runs.append(run)
        if record.current_run_id is None:
            record.current_run_id = run_id
        record.status = "queued"
        self.save()
        return run
