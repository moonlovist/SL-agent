from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core.config import VisionConfig
from .core.state import ExpertState
from .core.workflow import ExpertWorkflow
from .schemas.observation import ObservationDocument
from .vision.foreground_review import (
    apply_foreground_override,
    load_foreground_override,
    review_foreground_with_napari,
)
from .vision.pipeline import VisionPipeline
from .vision.preprocess import ImageData, PreprocessedImage, load_image, preprocess


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _load_saved_preprocessing(
    source: Path,
    image: ImageData,
    observation: ObservationDocument,
) -> PreprocessedImage | None:
    """Restore the completed run's preprocessing for an immediate review UI.

    The accepted annotation still triggers a complete fresh pipeline run.  This
    cache is used only to seed the editor with the exact automatic result the
    human is reviewing, avoiding another expensive fit before the window opens.
    """
    arrays_path = source / "vision_artifacts.npz"
    if not arrays_path.is_file():
        return None
    try:
        with np.load(arrays_path, allow_pickle=False) as saved:
            required = {
                "model", "residual", "residual_snr", "source_mask", "fit_mask",
                "initial_segmentation", "nucleus_yx",
            }
            if not required.issubset(saved.files):
                return None
            if tuple(saved["model"].shape) != tuple(image.data.shape):
                return None
            foreground = observation.foreground_galaxy
            quality = observation.data_quality
            nucleus = tuple(int(value) for value in saved["nucleus_yx"])
            if len(nucleus) != 2:
                return None
            return PreprocessedImage(
                image=image,
                model=np.asarray(saved["model"], dtype=float),
                residual=np.asarray(saved["residual"], dtype=float),
                residual_snr=np.asarray(saved["residual_snr"], dtype=float),
                source_mask=np.asarray(saved["source_mask"], dtype=bool),
                fit_mask=np.asarray(saved["fit_mask"], dtype=bool),
                segmentation=np.asarray(saved["initial_segmentation"], dtype=np.int32),
                nucleus_yx=(nucleus[0], nucleus[1]),
                background_median=float(quality.background_median),
                background_rms=float(quality.background_rms),
                residual_median=float(quality.residual_median),
                residual_robust_rms=float(quality.residual_robust_rms),
                model_method=quality.model_method,
                model_success=bool(quality.model_success),
                foreground_detected=bool(foreground.detected),
                foreground_peak_value=float(foreground.peak_value or 0.0),
                foreground_integrated_flux=foreground.integrated_flux,
                foreground_ellipticity=foreground.ellipticity,
                foreground_position_angle_deg=foreground.position_angle_deg,
                foreground_axis_ratio=foreground.axis_ratio,
                foreground_reliability=float(foreground.reliability),
                multiple_core_count=None,
                tidal_asymmetry=None,
                core_residual_fraction=None,
                flags=list(quality.flags),
                center_selection=dict(foreground.center_selection),
                source_protection_weight=(
                    np.asarray(saved["source_protection_weight"], dtype=float)
                    if "source_protection_weight" in saved.files else None
                ),
                human_foreground_seed=(
                    np.asarray(saved["human_foreground_seed"], dtype=bool)
                    if "human_foreground_seed" in saved.files else None
                ),
                human_fit_exclude=(
                    np.asarray(saved["human_fit_exclude"], dtype=bool)
                    if "human_fit_exclude" in saved.files else None
                ),
                human_analysis_exclude=(
                    np.asarray(saved["human_analysis_exclude"], dtype=bool)
                    if "human_analysis_exclude" in saved.files else None
                ),
            )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _reason(observation: ObservationDocument, output_dir: Path, *, mode: str = "observation_replay") -> ExpertState:
    state = ExpertState(
        image_path=observation.input_path,
        object_id=observation.object_id,
        observation_document=observation,
    )
    result = ExpertWorkflow().run(state)
    from .core.reproducibility import write_manifest
    _write_json(output_dir / "source_observation.json", observation.model_dump(mode="json"))
    _write_json(output_dir / "reasoning_observation.json", result.require_observation().model_dump(mode="json"))
    write_manifest(observation, output_dir, mode=mode, result=result)
    _write_json(output_dir / "reasoning.json", result.to_dict())
    _write_json(output_dir / "summary.json", result.final_decision)
    (output_dir / "report.md").write_text(
        result.final_report_markdown or "", encoding="utf-8"
    )
    return result


def analyze(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    object_id = input_path.stem if input_path.is_file() else input_path.name
    root = Path(args.output).expanduser().resolve()
    output_dir = root if args.flat_output else root / object_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config = VisionConfig(
        band=args.band,
        target_center_xy=(args.center_x, args.center_y) if args.center_x is not None and args.center_y is not None else None,
        ring_inner_px=args.ring_inner,
        ring_outer_px=args.ring_outer,
    )
    if (args.center_x is None) != (args.center_y is None):
        raise ValueError("Supply both --center-x and --center-y, or neither")
    foreground_override = None
    if args.foreground_override is not None or args.interactive_foreground:
        review_image = load_image(input_path, config.band)
        if args.foreground_override is not None:
            foreground_override = load_foreground_override(
                args.foreground_override, review_image
            )
        if args.interactive_foreground:
            suggested = preprocess(review_image, config)
            if foreground_override is not None:
                suggested = apply_foreground_override(
                    suggested,
                    foreground_override,
                    ring_outer_px=config.ring_outer_px,
                )
            foreground_override = review_foreground_with_napari(
                review_image,
                suggested,
                output_dir / "foreground_override.json",
                existing=foreground_override,
                ui_theme=args.ui_theme,
            )
    observation = VisionPipeline(config).run(
        input_path, output_dir=output_dir, foreground_override=foreground_override
    )
    _write_json(
        output_dir / "observation.json", observation.model_dump(mode="json")
    )
    result = _reason(observation, output_dir, mode="full_image_pipeline")
    print(f"Object: {observation.object_id}")
    print(f"Score status: {result.expert_assessment.score_status}")
    print(f"Vote category: {result.expert_assessment.vote_category}")
    print(f"Plausible vote categories: {list(result.expert_assessment.plausible_vote_categories)}")
    print(f"Decision audit passed: {result.decision_audit.audit_passed}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0 if result.decision_audit.audit_passed else 2


def foreground_review(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    prior_observation = source / "observation.json" if source.is_dir() else None
    if prior_observation is not None and prior_observation.exists():
        previous = ObservationDocument.model_validate_json(
            prior_observation.read_text(encoding="utf-8")
        )
        input_path = Path(previous.input_path).expanduser().resolve()
        band = previous.band
    else:
        input_path = source
        band = args.band
    object_id = input_path.stem if input_path.is_file() else input_path.name
    output_dir = Path(args.output).expanduser().resolve() / object_id
    output_dir.mkdir(parents=True, exist_ok=True)
    config = VisionConfig(
        band=band,
        ring_inner_px=args.ring_inner,
        ring_outer_px=args.ring_outer,
    )
    image = load_image(input_path, config.band)
    suggested = (
        _load_saved_preprocessing(source, image, previous)
        if prior_observation is not None and prior_observation.exists()
        else None
    )
    if suggested is None:
        print("Preparing automatic foreground suggestion...", flush=True)
        suggested = preprocess(image, config)
    else:
        print("Loaded saved foreground suggestion; opening editor...", flush=True)
    existing_path = (
        Path(args.foreground_override).expanduser().resolve()
        if args.foreground_override is not None
        else source / "foreground_override.json"
        if source.is_dir() and (source / "foreground_override.json").exists()
        else None
    )
    existing = (
        load_foreground_override(existing_path, image)
        if existing_path is not None
        else None
    )
    if existing is not None:
        suggested = apply_foreground_override(
            suggested,
            existing,
            ring_outer_px=config.ring_outer_px,
        )
    override = review_foreground_with_napari(
        image,
        suggested,
        output_dir / "foreground_override.json",
        existing=existing,
        ui_theme=args.ui_theme,
    )
    observation = VisionPipeline(config).run(
        input_path, output_dir=output_dir, foreground_override=override
    )
    _write_json(output_dir / "observation.json", observation.model_dump(mode="json"))
    result = _reason(observation, output_dir, mode="human_foreground_rerun")
    print(f"Object: {observation.object_id}")
    print(f"Vote category: {result.expert_assessment.vote_category}")
    print(f"Decision audit passed: {result.decision_audit.audit_passed}")
    print(f"Human override: {output_dir / 'foreground_override.json'}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0 if result.decision_audit.audit_passed else 2


def reason(args: argparse.Namespace) -> int:
    path = Path(args.observation).expanduser().resolve()
    observation = ObservationDocument.model_validate_json(path.read_text(encoding="utf-8"))
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _reason(observation, output_dir)
    print(f"Score status: {result.expert_assessment.score_status}")
    print(f"Vote category: {result.expert_assessment.vote_category}")
    print(f"Plausible vote categories: {list(result.expert_assessment.plausible_vote_categories)}")
    print(f"Decision audit passed: {result.decision_audit.audit_passed}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0 if result.decision_audit.audit_passed else 2


def desktop_ui(args: argparse.Namespace) -> int:
    from .ui.app import main as run_ui

    return run_ui(["--project", args.project])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lens-agent",
        description="Evidence-first strong gravitational lens expert agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run vision measurement and expert reasoning"
    )
    analyze_parser.add_argument("--center-x", type=float, default=None, help="Explicit target x pixel for off-centre cutouts")
    analyze_parser.add_argument("--center-y", type=float, default=None, help="Explicit target y pixel for off-centre cutouts")
    analyze_parser.add_argument("input", help="FITS/PNG file or target directory")
    analyze_parser.add_argument("--output", default="outputs", help="Output root")
    analyze_parser.add_argument("--flat-output", action="store_true")
    analyze_parser.add_argument("--band", default="VIS")
    analyze_parser.add_argument("--ring-inner", type=float, default=4.0)
    analyze_parser.add_argument("--ring-outer", type=float, default=25.0)
    analyze_parser.add_argument(
        "--interactive-foreground", action="store_true",
        help="Open a native-pixel foreground centre/mask editor before fitting",
    )
    analyze_parser.add_argument(
        "--foreground-override", default=None,
        help="Reuse a saved human foreground_override.json without opening the UI",
    )
    analyze_parser.add_argument(
        "--ui-theme", choices=("dark", "light"), default="dark",
        help="Theme for the optional foreground editor",
    )
    analyze_parser.set_defaults(handler=analyze)

    review_parser = subparsers.add_parser(
        "foreground-review",
        help="Edit a foreground selection and rerun from an input or prior result",
    )
    review_parser.add_argument("source", help="Input FITS/directory or prior object output directory")
    review_parser.add_argument("--output", default="outputs/human_foreground_rerun")
    review_parser.add_argument("--foreground-override", default=None)
    review_parser.add_argument("--band", default="VIS")
    review_parser.add_argument("--ring-inner", type=float, default=4.0)
    review_parser.add_argument("--ring-outer", type=float, default=25.0)
    review_parser.add_argument("--ui-theme", choices=("dark", "light"), default="dark")
    review_parser.set_defaults(handler=foreground_review)

    reason_parser = subparsers.add_parser(
        "reason", help="Run the expert layer from an Observation JSON only"
    )
    reason_parser.add_argument("observation")
    reason_parser.add_argument("--output", default="outputs/reasoning_only")
    reason_parser.set_defaults(handler=reason)
    ui_parser = subparsers.add_parser(
        "ui", help="Open the desktop batch-review application"
    )
    ui_parser.add_argument(
        "--project",
        default=str(Path.cwd() / "lensagent_ui_outputs"),
        help="UI project/output directory",
    )
    ui_parser.set_defaults(handler=desktop_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:
        parser.exit(1, f"lens-agent failed: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
