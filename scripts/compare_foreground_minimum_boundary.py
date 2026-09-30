from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def _find_foreground_bank(value):
    if isinstance(value, dict):
        if "source_boundary" in value and "selected_method" in value:
            return value
        for child in value.values():
            found = _find_foreground_bank(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_foreground_bank(child)
            if found is not None:
                return found
    return None


def _grade(path: Path) -> str:
    reasoning = json.loads((path / "reasoning.json").read_text(encoding="utf-8"))
    return str(reasoning["expert_assessment"]["vote_category"])


def _stretch(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    scale = max(float(np.percentile(finite, 99.5)), 1e-8)
    return np.arcsinh(np.clip(values, 0, None) / scale * 20.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("object_ids", nargs="+")
    args = parser.parse_args()

    figure, axes = plt.subplots(len(args.object_ids), 6, figsize=(20, 6.4 * len(args.object_ids)))
    axes = np.atleast_2d(axes)
    for row, object_id in enumerate(args.object_ids):
        old_path = args.old_root / object_id
        new_path = args.new_root / object_id
        old = np.load(old_path / "vision_artifacts.npz")
        new = np.load(new_path / "vision_artifacts.npz")
        observation = json.loads((new_path / "observation.json").read_text(encoding="utf-8"))
        bank = _find_foreground_bank(observation) or {}
        boundary = bank.get("source_boundary", {})
        cy, cx = map(int, new["nucleus_yx"])
        half = 45
        y0, y1 = max(0, cy - half), min(new["data"].shape[0], cy + half + 1)
        x0, x1 = max(0, cx - half), min(new["data"].shape[1], cx + half + 1)
        sl = np.s_[y0:y1, x0:x1]
        panels = [
            (_stretch(new["data"][sl]), "Input + new boundary", "gray", None),
            (_stretch(old["model"][sl]), f"Old foreground model\nold grade {_grade(old_path)}", "gray", None),
            (old["residual_snr"][sl], "Old residual S/N", "coolwarm", (-5, 5)),
            (_stretch(new["model"][sl]), f"New foreground model\nnew grade {_grade(new_path)}", "gray", None),
            (new["residual_snr"][sl], "New residual S/N", "coolwarm", (-5, 5)),
            (new["source_protection_weight"][sl], "New fit inclusion\n1=fit, 0=exclude", "viridis", (0, 1)),
        ]
        for column, (values, title, cmap, limits) in enumerate(panels):
            kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
            axes[row, column].imshow(values, origin="lower", cmap=cmap, **kwargs)
            axes[row, column].set_title(title)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        if boundary.get("boundary_detected"):
            radius = float(boundary["boundary_radius_px"])
            q = float(boundary["boundary_axis_ratio_q"])
            angle = float(boundary["boundary_position_angle_deg"])
            axes[row, 0].add_patch(Ellipse(
                (cx - x0, cy - y0), 2 * radius, 2 * radius * q,
                angle=angle, fill=False, edgecolor="cyan", linewidth=2,
            ))
            note = (
                f"minimum boundary: r={radius:.1f}px, "
                f"support={boundary['selected_angular_support_deg']:.0f}deg"
            )
        else:
            note = "no supported two-sided minimum: no boundary imposed"
        axes[row, 0].set_ylabel(f"{object_id}\n{note}", fontsize=10)

    figure.suptitle("Foreground subtraction before/after v0.26.0", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    main()
