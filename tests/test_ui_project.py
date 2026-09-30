import json
from pathlib import Path
from uuid import uuid4

from expert_agent.ui.project import (
    ProjectIndex,
    discover_inputs,
    infer_reference_grade,
    load_result_snapshot,
    render_input_preview,
)


def test_reference_grade_is_only_inferred_from_explicit_label_directory() -> None:
    assert infer_reference_grade(Path("data") / "A" / "object" / "object.fits")[0] == "A"
    assert infer_reference_grade(Path("data") / "B样本" / "object.fits")[0] == "B"
    assert infer_reference_grade(Path("data") / "beauty" / "object.fits") == (None, None)


def test_reference_grade_confirmation_persists() -> None:
    tmp_path = _test_dir()
    source = tmp_path / "beauty" / "sample.fits"
    source.parent.mkdir()
    source.write_bytes(b"fits")
    project = ProjectIndex(tmp_path / "project")
    record = project.add_inputs([source])[0]
    assert record.reference_grade is None
    assert not record.reference_grade_confirmed
    record.reference_grade = "B"
    record.reference_grade_source = "人工标定"
    record.reference_grade_confirmed = True
    project.save()

    restored = ProjectIndex(project.root).objects["sample"]
    assert restored.reference_grade == "B"
    assert restored.reference_grade_source == "人工标定"
    assert restored.reference_grade_confirmed


def test_ui_chinese_assessment_translation_preserves_measurements() -> None:
    from expert_agent.ui.app import _cn_evidence, _cn_rationale

    rationale = _cn_rationale(
        "The candidate has a lens-like morphology, but its system-level specificity "
        "is incomplete or a non-lens explanation remains comparably plausible."
    )
    evidence = _cn_evidence(
        "The measured scene contains 3 promoted structures; common-radius CV=0.11, "
        "total angular coverage=61.17 deg, and mean tangential misalignment=6.08."
    )

    assert "透镜样形态" in rationale
    assert "3 个晋级结构" in evidence
    assert "61.17°" in evidence
    assert "6.08°" in evidence


def _test_dir() -> Path:
    path = Path.cwd() / ".codex_tmp" / "ui-project-tests" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_discover_inputs_prefers_object_named_fits() -> None:
    tmp_path = _test_dir()
    first = tmp_path / "object_a"
    first.mkdir()
    preferred = first / "object_a.fits"
    preferred.write_bytes(b"fits")
    (first / "auxiliary.fits").write_bytes(b"aux")
    second = tmp_path / "loose.fits"
    second.write_bytes(b"loose")

    found = discover_inputs(tmp_path)

    assert preferred.resolve() in found
    assert second.resolve() in found
    assert (first / "auxiliary.fits").resolve() not in found


def test_load_result_snapshot_maps_scientific_groups() -> None:
    tmp_path = _test_dir()
    diagnostic = tmp_path / "diagnostic.png"
    diagnostic.write_bytes(b"png")
    summary = {
        "object_id": "sample",
        "published_grade": "B",
        "expert_assessment": {
            "vote_category": "B",
            "assessment_rationale": "A probable lens configuration remains.",
            "assessment_questions": {"colour_evidence_state": "usable"},
        },
        "evidence_dossier": {
            "supporting_evidence": [{"statement": "Tangential structure."}],
            "contradicting_evidence": [{"statement": "Subtraction sensitive."}],
            "processing_cautions": [],
        },
        "decision_audit": {
            "audit_passed": True,
            "abstention_recommended": False,
            "required_corrections": [],
        },
    }
    observation = {
        "object_id": "sample",
        "input_path": str(tmp_path / "sample.fits"),
        "artifacts": {"diagnostic_image": str(diagnostic)},
        "data_quality": {
            "usable": True,
            "model_success": True,
            "flags": [],
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "observation.json").write_text(json.dumps(observation), encoding="utf-8")

    snapshot = load_result_snapshot(tmp_path)

    assert snapshot.grade == "B"
    assert snapshot.group == "B"
    assert not snapshot.needs_review
    assert snapshot.supporting == ["Tangential structure."]


def test_project_index_persists_versioned_runs() -> None:
    tmp_path = _test_dir()
    source = tmp_path / "sample.fits"
    source.write_bytes(b"fits")
    project = ProjectIndex(tmp_path / "project")
    record = project.add_inputs([source])[0]
    run = project.new_run(record, kind="auto")
    run.status = "completed"
    run.grade = "A"
    record.status = "completed"
    project.save()

    restored = ProjectIndex(project.root)

    assert restored.objects["sample"].current_run().grade == "A"
    assert restored.objects["sample"].current_run().output_dir.endswith("sample")


def test_project_index_syncs_external_reasoning_grade() -> None:
    tmp_path = _test_dir()
    source = tmp_path / "sample.fits"
    source.write_bytes(b"fits")
    project = ProjectIndex(tmp_path / "project")
    record = project.add_inputs([source])[0]
    run = project.new_run(record, kind="auto")
    output = Path(run.output_dir)
    output.mkdir(parents=True)
    (output / "observation.json").write_text(
        json.dumps({
            "object_id": "sample",
            "input_path": str(source),
            "artifacts": {},
            "data_quality": {"usable": True, "model_success": True, "flags": []},
        }),
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps({
            "object_id": "sample",
            "published_grade": "A",
            "expert_assessment": {},
            "evidence_dossier": {},
            "decision_audit": {},
        }),
        encoding="utf-8",
    )
    run.status = "completed"
    run.grade = "C"
    record.status = "completed"
    project.save()

    restored = ProjectIndex(project.root)

    assert restored.objects["sample"].current_run().grade == "A"


def test_render_input_preview_preserves_native_field() -> None:
    from astropy.io import fits
    import numpy as np
    from PIL import Image

    root = _test_dir()
    source = root / "preview.fits"
    data = np.arange(48, dtype=np.float32).reshape(6, 8)
    fits.PrimaryHDU(data).writeto(source)

    preview = render_input_preview(source, root / "cache")
    rendered = np.asarray(Image.open(preview))

    assert rendered.shape == data.shape
    assert rendered[0, -1] > rendered[-1, -1]
