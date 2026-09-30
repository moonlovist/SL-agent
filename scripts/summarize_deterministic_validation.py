"""Summarize frozen local regression outputs without rerating any object."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs_deterministic_review_20260907"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


before = read(OUT / "before/results.json")
after = read(OUT / "final/results.json")
prior = {r["object_id"]: r for r in before}
changes = [dict(object_id=r["object_id"], before=prior[r["object_id"]]["vote"], after=r["vote"])
           for r in after if r["vote"] != prior[r["object_id"]]["vote"]]
full = []
for path in sorted((OUT / "full_fits").glob("*/summary.json")):
    result = read(path)
    full.append(dict(object_id=path.parent.name, published=result["published_grade"],
                     audit=result["decision_audit"]["audit_passed"], report=str(path.parent / "report.md")))

object_id = "102021493_NEG586398445479765591"
first = OUT / "full_fits" / object_id
second = OUT / "full_fits_repeat" / object_id
one = read(first / "observation.json")
two = read(second / "observation.json")
with np.load(one["artifacts"]["vision_arrays"], allow_pickle=False) as a, np.load(two["artifacts"]["vision_arrays"], allow_pickle=False) as b:
    matching = {key: np.array_equal(a[key], b[key], equal_nan=True) if a[key].dtype.kind in "fc" else np.array_equal(a[key], b[key]) for key in a.files}
    array_keys_equal = set(a.files) == set(b.files)
measurement_equal = {key: one[key] == two[key] for key in [
    "measurements", "structures", "model_candidates", "candidate_handoff_review", "data_quality", "scene_geometry", "ring_candidates"]}
grade_equal = read(first / "summary.json")["expert_assessment"] == read(second / "summary.json")["expert_assessment"]
fingerprints = {read(p)["code_sha256"] for p in (OUT / "final").glob("*/run_manifest.json")}
fingerprints.update(read(p)["code_sha256"] for p in (OUT / "full_fits").glob("*/run_manifest.json"))
test = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
(OUT / "pytest.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
validation = dict(changes=changes, replay_count=len(after),
    replay_repeat_equal=all(r["repeat_equal"] for r in after),
    all_replay_audits_passed=all(r["audit"] for r in after), full_fits=full,
    full_repeat=dict(object_id=object_id, keys_equal=array_keys_equal, arrays_equal=matching,
                     measurements_equal=measurement_equal, assessment_equal=grade_equal),
    code_fingerprints=sorted(fingerprints), pytest_exit_code=test.returncode)
(OUT / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

lines = ["# v0.21 确定性实验版：修改与验证结果", "",
         "本版不接入多模态。FITS → 像素测量 → 证据合并 → 一次固定规则裁决 → 重放审计 → 发布等级。", "",
         "这不是 Q1 独立盲测：标签沿用用户提供的 A/B 文件夹，未重新按官方目录交叉核对。", "",
         "## 保存像素重测与评级回放（25 个）", "",
         "这一步使用原先保存的像素数组，不等于 25 个全部从 FITS 重新跑前处理。修改前和修改后均单独留档。", "",
         "| 对照组 | 修改前 | 修改后 |", "|---|---|---|"]
for label in ["A", "B"]:
    counts = [dict(Counter(r["vote"] for r in rows if r["expected"] == label)) for rows in [before, after]]
    lines.append(f"| {label} | {counts[0]} | {counts[1]} |")
lines += ["", "insufficient_for_grading 对外发布为 unknown。", "",
          "## 等级发生变化的对象", "", "| 对象 | 修改前 | 修改后 |", "|---|---|---|"]
for row in changes:
    path = OUT / "final" / row["object_id"] / "report.md"
    lines.append(f"| [{row['object_id']}](<{path.as_posix()}>) | {row['before']} | {row['after']} |")
lines += ["", "## 原始 FITS 完整运行（4 个）", "", "| 对象 | 新等级 | 审计 |", "|---|---|---|"]
for row in full:
    lines.append(f"| [{row['object_id']}](<{Path(row['report']).as_posix()}>) | {row['published']} | {row['audit']} |")
lines += ["", "## 可复现验证", "",
          f"- 单元/回归测试：退出码 {test.returncode}；完整输出见 [pytest.txt](pytest.txt)。",
          f"- 25 个保存测量重复裁决：{validation['replay_repeat_equal']}；全部审计通过：{validation['all_replay_audits_passed']}。",
          f"- 5591 从原始 FITS 再跑一次：数组集合一致 {array_keys_equal}，{sum(matching.values())}/{len(matching)} 个数组逐值一致（NaN 同位视为相同）。",
          f"- 该完整重跑的测量字段一致：{all(measurement_equal.values())}；最终内部判断一致：{grade_equal}。",
          f"- final 回放与完整 FITS 的代码指纹数量：{len(fingerprints)}（应为 1）。",
          "- 运行时间戳与输出路径预期不同，不纳入数值一致性比较；不承诺跨平台逐位一致。", "",
          "## 结论与局限", "",
          "A51–70 的四个 B 变为 A；9454 仍为 B。B 前五个未改善：3A、1B、1unknown。说明语义/交接修正生效，但 A 的纯度和 B/C 界限尚未解决，不能宣称科学性能已全面提升。", "",
          "确定性和可追溯验证通过，不等于复现人类专家判断的准确率验证通过。SNR/相关噪声、前景光系统误差、多波段和替代解释仍需要独立数据验证。", "",
          "[逐步流程、科学依据与未解决项](<../../src/docs/DETERMINISTIC_V0210_CN.md>)", "",
          "原输出未覆盖。before/source_snapshot 保留修改前源码；final 和 full_fits 内保留运行时源码、指纹、参数和报告。旧候选二次评级实现已移除，可由修改前源码快照恢复。"]
(OUT / "RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps(validation, ensure_ascii=False, indent=2))
