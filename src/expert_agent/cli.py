from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.config import VisionConfig
from .core.state import ExpertState
from .core.workflow import ExpertWorkflow
from .schemas.observation import ObservationDocument
from .vision.pipeline import VisionPipeline


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _reason(observation: ObservationDocument, output_dir: Path) -> ExpertState:
    state = ExpertState(
        image_path=observation.input_path,
        object_id=observation.object_id,
        observation_document=observation,
    )
    result = ExpertWorkflow().run(state)
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
        ring_inner_px=args.ring_inner,
        ring_outer_px=args.ring_outer,
    )
    observation = VisionPipeline(config).run(input_path, output_dir=output_dir)
    _write_json(
        output_dir / "observation.json", observation.model_dump(mode="json")
    )
    result = _reason(observation, output_dir)
    print(f"Object: {observation.object_id}")
    print(f"Classification: {result.final_decision['classification']}")
    print(
        f"Confidence: {result.confidence_assessment.level} "
        f"({result.confidence_assessment.score:.3f})"
    )
    print(f"Audit valid: {result.audit_result.valid}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


def reason(args: argparse.Namespace) -> int:
    path = Path(args.observation).expanduser().resolve()
    observation = ObservationDocument.model_validate_json(path.read_text(encoding="utf-8"))
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = _reason(observation, output_dir)
    print(f"Classification: {result.final_decision['classification']}")
    print(f"Report: {output_dir / 'report.md'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lens-agent",
        description="Evidence-first strong gravitational lens expert agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run vision measurement and expert reasoning"
    )
    analyze_parser.add_argument("input", help="FITS/PNG file or target directory")
    analyze_parser.add_argument("--output", default="outputs", help="Output root")
    analyze_parser.add_argument("--flat-output", action="store_true")
    analyze_parser.add_argument("--band", default="VIS")
    analyze_parser.add_argument("--ring-inner", type=float, default=4.0)
    analyze_parser.add_argument("--ring-outer", type=float, default=25.0)
    analyze_parser.set_defaults(handler=analyze)

    reason_parser = subparsers.add_parser(
        "reason", help="Run the expert layer from an Observation JSON only"
    )
    reason_parser.add_argument("observation")
    reason_parser.add_argument("--output", default="outputs/reasoning_only")
    reason_parser.set_defaults(handler=reason)
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
