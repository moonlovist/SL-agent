import os
import json
from typing import Optional

from .state import SpectroState


class ResultWriter:
    """
    ResultWriter
    -------------
    将一次 workflow 的 SpectroState 写成可消费的结果文件。
    - 只读 state
    - 不修改分析逻辑
    - 可安全在循环中重复调用
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        overwrite: bool = True,
        encoding: str = "utf-8"
    ):
        self.output_dir = output_dir
        self.overwrite = overwrite
        self.encoding = encoding

    def write_redshift_hypotheses(self, state: SpectroState) -> None:
        """
        将 redshift_hypotheses 写出为 .txt 文件。
        文件名格式：
          {file_name}_redshift_hypotheses.txt
        """
        output_dir = self._resolve_output_dir(state)
        os.makedirs(output_dir, exist_ok=True)
        file_name = state.get("file_name", "unknown")

        data = state.get('redshift_hypotheses')
        if not data:
            return

        path = os.path.join(output_dir, f"{file_name}_redshift_hypotheses.txt")
        with open(path, "w", encoding=self.encoding) as f:
            f.write(f"{'='*60}\n  REDSHIFT HYPOTHESES\n{'='*60}\n\n")
            # new format: dict with z, zmedian, hypotheses
            if isinstance(data, dict):
                f.write(f"zmedian: {data.get('zmedian')}\n")
                z_list = data.get('z', [])
                f.write(f"z ({len(z_list)} values): {z_list}\n\n")
                hypotheses = data.get('hypotheses', [])
            else:
                # backwards-compat: old list format
                hypotheses = data

            # ── 跳过 verbose 字段，只打印关键字段 ──
            _key_order = [
                "Hypothesis", "z_representative", "score", "z_center",
                "z_list", "z_max", "z_min", "z_spread",
                "N_emission", "N_absorption",
                "Emission matches", "Absorption matches",
                "scoring_details", "matched_lines",
            ]
            for idx, entry in enumerate(hypotheses, start=1):
                f.write(f"--- Match #{idx} ---\n")
                for key in _key_order:
                    if key in entry:
                        f.write(f"  {key}: {entry[key]}\n")
                # 其余字段
                for key, value in entry.items():
                    if key not in _key_order:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")

    def write_hypothesis_analysis(self, state: SpectroState) -> None:
        """Write harness-based rule analysis results."""
        output_dir, file_name = self._resolve_output_dir(state), state.get("file_name", "unknown")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{file_name}_hypothesis_analysis.txt")

        synthesis = state.get('hypothesis_analysis') or {}
        results = state.get('hypothesis_results') or []

        with open(path, "w", encoding=self.encoding) as f:
            f.write("=" * 60 + "\n")
            f.write("  SYNTHESIS VERDICT\n")
            f.write("=" * 60 + "\n")
            f.write(json.dumps(synthesis, indent=2, ensure_ascii=False))
            f.write("\n\n")

            f.write("=" * 60 + "\n")
            f.write("  PER-HYPOTHESIS REPORTS\n")
            f.write("=" * 60 + "\n")
            for r in results:
                f.write(f"\n--- Hypothesis {r.get('hypothesis_idx','?')} "
                        f"(z={r.get('redshift','?')}) ---\n")
                f.write(r.get('report', '(no report)'))
                f.write("\n")

    def write(self, state: SpectroState) -> None:
        """Write all result files for a completed analysis run."""
        self.write_redshift_hypotheses(state)
        self.write_hypothesis_analysis(state)

    # =========================
    # 🧰 Helpers
    # =========================

    def _resolve_output_dir(self, state: SpectroState) -> str:
        return self.output_dir or state.get("output_dir") or "outputs"
