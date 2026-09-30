from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import argparse
from pathlib import Path
import re
import sys

from .project import (
    GROUP_ORDER,
    ObjectRecord,
    ProjectIndex,
    ResultSnapshot,
    RunRecord,
    discover_inputs,
    infer_reference_grade,
    load_result_snapshot,
    object_id_from_input,
    render_input_preview,
    utc_now,
)
from .theme import PALETTE, stylesheet
from ..core.llm_client import LLMConfig, failure_message


CORE_ARTIFACTS = (
    ("input_rgb", "原始 RGB"),
    ("input_vis", "原始 VIS（线性）"),
    ("rgb_context_diagnostic", "RGB 定位审计"),
    ("target_selection_diagnostic", "目标与前景选择"),
    ("foreground_model_comparison_diagnostic", "前景光模型比较"),
    ("diagnostic_image", "前景减除与结构候选"),
    ("signal_audit_diagnostic", "信号审计"),
    ("source_boundary_diagnostic", "源像保护边界"),
)

RESULT_FILES = (
    ("llm_assessment.json", "DeepSeek 假设与裁决"),
    ("llm_input.json", "DeepSeek 使用的观测测量"),
    ("observation_summary_cn.md", "中文观测摘要"),
    ("report.md", "完整评审报告"),
    ("summary.json", "判定结果 summary.json"),
    ("observation.json", "观测结果 observation.json"),
    ("reasoning.json", "推理记录 reasoning.json"),
    ("vision_manifest.json", "图像清单 vision_manifest.json"),
)

@dataclass
class JobSpec:
    object_id: str
    run_id: str
    program: str
    arguments: list[str]
    label: str


def _qt():
    try:
        from qtpy import QtCore, QtGui, QtWidgets
    except ImportError as exc:
        raise RuntimeError(
            'LensAgent UI requires the "ui" optional dependency: pip install -e ".[ui]"'
        ) from exc
    return QtCore, QtGui, QtWidgets


QtCore, QtGui, QtWidgets = _qt()
Signal = QtCore.Signal
ROLE_KIND = int(QtCore.Qt.ItemDataRole.UserRole)
ROLE_OBJECT = ROLE_KIND + 1
ROLE_PATH = ROLE_KIND + 2
ROLE_TITLE = ROLE_KIND + 3


RATIONALE_CN = {
    "Unresolved signed foreground-model conflict.": "不同前景模型给出了符号相反的残差结果，当前解释仍未解决。",
    "Observation usability or essential noise integrity is insufficient; the expert abstains rather than assigning X.": "观测可用性或关键噪声信息不足，因此暂不判级，而不是直接判为 X。",
    "At least one supported lens configuration is clear in the available image evidence, and no equally strong measured non-lens explanation remains.": "现有图像证据中至少有一组得到支持的清晰透镜构型，且没有同等强度的已测量非透镜解释。",
    "Lens-like morphology and an equally strong measured alternative coexist: possible, not probable.": "透镜样形态与同等强度的其他已测量解释同时存在，因此只能视为可能透镜，而非高概率透镜。",
    "Lens-like morphology remains available, but opposite-signed subtraction results leave its interpretation unresolved; this is possible-lens evidence, not confirmed contamination or a probable-lens claim.": "仍存在透镜样形态，但符号相反的减除结果使其解释未定；这只能支持可能透镜，不能确认污染，也不足以宣称高概率透镜。",
    "The morphology is clearly lens-like, but quality-controlled colours are strongly inconsistent between proposed images after allowing an achromatic magnification offset. The system remains probable rather than high-confidence A.": "形态清晰地支持透镜解释，但在允许无色放大差异后，候选像之间经过质量控制的颜色仍明显不一致，因此保留为 B，而不是高置信度 A。",
    "Lens-specific morphology favors lensing, but subtraction sensitivity remains unresolved. Comparison failure does not erase the structure or establish contamination.": "透镜特异性形态倾向于透镜解释，但对前景减除的敏感性尚未解决；比较模型失败既不能抹除该结构，也不能证明其受到污染。",
    "Credible structures have directional or relational lens specificity, but the current image evidence does not yet form a clear high-confidence configuration.": "可信结构具有方向性或相互关系上的透镜特征，但当前图像证据尚未形成清晰的高置信度构型。",
    "Measurable geometry-related held candidates survive comparison, but threshold persistence or extraction is unresolved. They justify only a possible-lens C vote, not X, B, or A.": "可测量的几何相关候选在比较模型中仍然存在，但阈值持续性或提取结果尚未解决，因此只足以支持可能透镜 C，而不是 X、B 或 A。",
    "The candidate has a lens-like morphology, but its system-level specificity is incomplete or a non-lens explanation remains comparably plausible.": "候选体具有透镜样形态，但系统层面的透镜特异性仍不完整，或仍存在同样合理的非透镜解释。",
    "The usable Observation does not contain a credible lens-specific structural relationship and is better described by non-lens or noise explanations.": "可用观测中没有可信的透镜特异性结构关系，现有证据更适合由非透镜结构或噪声解释。",
    "A probable lens configuration remains.": "仍存在一组较可能的透镜构型。",
}

COLOUR_STATE_CN = {
    "not_measured": "未测量",
    "foreground_ambiguous": "与前景颜色关系存在歧义",
    "foreground_distinct_support": "候选与前景颜色可区分，提供辅助支持",
    "no_testable_same_source_pair": "没有可检验的同源候选像对",
    "mixed": "不同候选像对的颜色结论混合",
    "same_source_conflict": "候选同源像之间存在颜色冲突",
    "supports_common_source": "颜色关系支持共同源",
    "partially_supports_common_source": "颜色仅覆盖部分必要多像，提供部分同源支持",
    "same_source_colour_tension": "候选同源像存在颜色张力，但尚未达到可靠冲突",
    "inconclusive": "颜色结果不确定",
    "suggestive_consistent_with_achromatic_scaling": "中等可信颜色提示可能同源（不参与升降级）",
    "suggestive_colour_conflict": "中等可信颜色存在提示性差异（不参与升降级）",
    "suggestive_inconclusive": "中等可信颜色仍不确定（不参与升降级）",
    "suggestive_mixed_colour_relation": "多组中等可信颜色结论混合（不参与升降级）",
    "low_snr_flux_shape_diagnostic_only": "仅有低信噪比通量形状诊断",
    "unknown": "未知",
    "usable": "可用",
}


def _cn_value(value: str) -> str:
    return {"True": "是", "False": "否", "None": "未测得", "unknown": "未知"}.get(value, value)


def _cn_rationale(text: str) -> str:
    if not text:
        return "未提供判定摘要。"
    return RATIONALE_CN.get(text, text if re.search(r"[\u4e00-\u9fff]", text) else "该判定摘要尚无中文展示文本，请查看完整报告。")


def _cn_evidence(text: str) -> str:
    if not text:
        return "无"
    if re.search(r"[\u4e00-\u9fff]", text):
        return text
    match = re.match(
        r"Structure (.+?) has integrated S/N=(.+?), radius=(.+?) px, angular extent=(.+?) deg, "
        r"tangential misalignment=(.+?) deg, and near_core=(.+?)\. Comparison threshold occupancy=(.+?); "
        r"comparison nominal aperture SNR=(.+?); pixel core overlap=(.+?); occupied angle=(.+?), "
        r"radial fractional width=(.+?), local tangential fraction=(.+?)\.$",
        text,
    )
    if match:
        sid, snr, radius, extent, misalignment, near_core, occupancy, comparison_snr, overlap, occupied, radial_width, local_tangent = match.groups()
        return (
            f"结构 {sid}：综合 S/N={snr}，半径={radius} 像素，角向跨度={extent}°，"
            f"切向偏差={misalignment}°，接近核心={_cn_value(near_core)}。"
            f"比较模型超阈值占比={occupancy}，比较模型名义孔径 S/N={_cn_value(comparison_snr)}，"
            f"核心像素重叠比例={overlap}，占据角度={occupied}°，径向宽度比例={radial_width}，"
            f"局部切向比例={_cn_value(local_tangent)}。"
        )
    match = re.match(
        r"The measured scene contains (.+?) promoted structures; common-radius CV=(.+?), "
        r"total angular coverage=(.+?) deg, and mean tangential misalignment=(.+)\.$", text,
    )
    if match:
        count, radius_cv, coverage, misalignment = match.groups()
        return f"测量场景包含 {count} 个晋级结构；共同半径变异系数={radius_cv}，总角向覆盖={coverage}°，平均切向偏差={misalignment}°。"
    match = re.match(
        r"A same-data foreground model comparison reports success=(.+?); threshold occupancy in the historical structure mask=(.+?) and residual correlation=(.+)\.$",
        text,
    )
    if match:
        success, occupancy, correlation = match.groups()
        return f"同一数据上的前景模型比较：拟合成功={_cn_value(success)}，历史结构掩膜中的超阈值占比={occupancy}，残差相关系数={correlation}。"
    match = re.match(r"Candidate (.+?) is supported by one geometric search route; that route is not independent lens evidence\.", text)
    if match:
        return f"候选 {match.group(1)} 得到一条几何搜索路线支持；该路线不是独立的透镜证据。"
    match = re.match(r"Candidate (.+?) entered through route recovery\. Recovery is provenance and is not itself negative evidence\.", text)
    if match:
        return f"候选 {match.group(1)} 由路线恢复通道进入；路线恢复只记录来源，本身不是负面证据。"
    match = re.match(r"Candidate (.+?) has core-risk context; measured pixel overlap=(.+?)\. Not an origin diagnosis\.", text)
    if match:
        return f"候选 {match.group(1)} 位于核心敏感区域；测得的像素重叠比例={match.group(2)}。这不能诊断其物理来源。"
    exact = {
        "Tangential structure.": "存在切向结构。",
        "Subtraction sensitive.": "该结构对前景减除较敏感。",
        "Primary foreground modelling failed; model dependence remains unresolved, not proven artifact.": "主前景模型拟合失败；模型依赖性仍未解决，但这不能证明结构是伪影。",
        "Relevant apertures have significantly negative nominal flux in the comparison residual; model dependence is unresolved, not proven contamination.": "相关孔径在比较模型残差中具有显著负的名义通量；模型依赖性尚未解决，但这不能证明污染。",
        "Usable candidate colours do not separate the structure from the foreground SED. This is non-discriminating rather than contradictory and therefore does not impose a grade cap.": "可用候选颜色无法将该结构与前景星系的光谱能量分布可靠区分；这属于无区分力，而非矛盾证据，因此不限制等级上限。",
        "A measured structure has the joint morphology of a core-crossing foreground disc residual. It is retained for audit but cannot by itself support an arc/ring configuration.": "一个已测结构同时具有穿过核心的前景盘残差形态；它会保留供审计，但不能单独支持弧或环构型。",
        "Pooled pixel survival is low; inspect structure-specific survival, nearby persistence and negative residuals. The pooled fraction alone is not a non-lens explanation.": "合并像素的存活比例较低；应分别检查各结构的存活、邻近持续性和负残差。仅凭合并比例不能构成非透镜解释。",
    }
    return exact.get(text, "该技术条目尚无中文展示文本，请在完整报告中查看原始记录。")


class BrandMark(QtWidgets.QWidget):
    """Compact SL-agent mark matching the reference title bar footprint."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)

    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor("#B7E0FA"))
        center = QtCore.QPointF(12.0, 12.0)
        painter.drawEllipse(center, 3.2, 3.2)
        for dx, dy in ((0, -8), (7, -4), (7, 4), (0, 8), (-7, 4), (-7, -4)):
            painter.drawEllipse(QtCore.QPointF(12.0 + dx, 12.0 + dy), 2.25, 2.25)


def _folder_icon(color: str = "#B7E0FA") -> QtGui.QIcon:
    """Draw a theme-safe monochrome folder icon without OS icon colours."""
    pixmap = QtGui.QPixmap(18, 18)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    pen = QtGui.QPen(QtGui.QColor(color), 1.6)
    pen.setJoinStyle(QtCore.Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
    path = QtGui.QPainterPath()
    path.moveTo(2.0, 5.5)
    path.lineTo(7.0, 5.5)
    path.lineTo(8.6, 7.0)
    path.lineTo(16.0, 7.0)
    path.lineTo(15.2, 14.5)
    path.lineTo(2.8, 14.5)
    path.closeSubpath()
    painter.drawPath(path)
    painter.drawLine(QtCore.QPointF(2.0, 5.5), QtCore.QPointF(2.8, 14.5))
    painter.end()
    return QtGui.QIcon(pixmap)


def _circle_icon(color: str) -> QtGui.QIcon:
    """Small solid status bullet for grade and queue group headings."""
    pixmap = QtGui.QPixmap(14, 14)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(color))
    painter.drawEllipse(QtCore.QRectF(2.0, 2.0, 10.0, 10.0))
    painter.end()
    return QtGui.QIcon(pixmap)


class ImageCanvas(QtWidgets.QLabel):
    """Large image canvas supporting fit, native size, and interactive zoom."""

    zoomRequested = Signal(float)
    scaleChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_path = ""
        self.source_pixmap = QtGui.QPixmap()
        self.fit_mode = True
        self.zoom_factor = 1.0
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("imageCanvas")
        self.setText("从左侧展开样本并选择一张过程图")

    def set_source(self, path: str) -> None:
        self.source_path = path
        self.source_pixmap = QtGui.QPixmap(path)
        if self.source_pixmap.isNull():
            self.setText("图像不可用")
            self.setPixmap(QtGui.QPixmap())
            return
        self.refresh_scale()

    def refresh_scale(self) -> None:
        if self.source_pixmap.isNull():
            return
        if self.fit_mode:
            target = self.parentWidget().size() if self.parentWidget() else self.size()
            pixmap = self.source_pixmap.scaled(
                max(1, target.width() - 28),
                max(1, target.height() - 28),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.zoom_factor = min(
                pixmap.width() / max(1, self.source_pixmap.width()),
                pixmap.height() / max(1, self.source_pixmap.height()),
            )
            self.setMinimumSize(1, 1)
            self.setPixmap(pixmap)
        else:
            scaled_size = QtCore.QSize(
                max(1, round(self.source_pixmap.width() * self.zoom_factor)),
                max(1, round(self.source_pixmap.height() * self.zoom_factor)),
            )
            pixmap = self.source_pixmap.scaled(
                scaled_size,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.setMinimumSize(pixmap.size())
            self.resize(pixmap.size())
            self.setPixmap(pixmap)
        self.scaleChanged.emit(self.zoom_factor)

    def set_zoom(self, factor: float) -> None:
        self.fit_mode = False
        self.zoom_factor = max(0.05, min(12.0, factor))
        self.updateGeometry()
        self.refresh_scale()

    def wheelEvent(self, event) -> None:
        if self.source_pixmap.isNull() or event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return
        self.zoomRequested.emit(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.fit_mode:
            self.refresh_scale()

    def sizeHint(self) -> QtCore.QSize:
        if self.fit_mode:
            return QtCore.QSize(1, 1)
        if self.source_pixmap.isNull():
            return QtCore.QSize(1, 1)
        return QtCore.QSize(
            max(1, round(self.source_pixmap.width() * self.zoom_factor)),
            max(1, round(self.source_pixmap.height() * self.zoom_factor)),
        )

    def minimumSizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(1, 1) if self.fit_mode else self.sizeHint()


class ImagePreview(QtWidgets.QDialog):
    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Path(path).name)
        self.resize(1100, 820)
        layout = QtWidgets.QVBoxLayout(self)
        area = QtWidgets.QScrollArea()
        area.setWidgetResizable(True)
        label = QtWidgets.QLabel()
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        pixmap = QtGui.QPixmap(path)
        label.setPixmap(pixmap)
        area.setWidget(label)
        layout.addWidget(area)


class ReferenceGradeDialog(QtWidgets.QDialog):
    """Assign one shared reference grade, with per-object overrides."""

    OPTIONS = (
        ("请选择", None),
        ("A", "A"),
        ("B", "B"),
        ("C", "C"),
        ("X", "X"),
        ("未提供", ""),
    )

    def __init__(self, items: list[tuple[str, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("标定 Euclid 原始等级")
        self.resize(680, min(650, 235 + 42 * max(1, len(items))))
        self.assignments: dict[str, str | None] = {}
        self._combos: dict[str, QtWidgets.QComboBox] = {}
        layout = QtWidgets.QVBoxLayout(self)
        explanation = QtWidgets.QLabel(
            "这些标签只用于和 Agent 结果对照，不会进入 Agent 的推理或评分。\n"
            "若所选文件属于同一等级，先选统一默认等级；个别不同的样本可在表格中单独修改。"
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        default_row = QtWidgets.QHBoxLayout()
        default_row.addWidget(QtWidgets.QLabel("统一默认等级："))
        self.default_combo = self._new_combo()
        self.default_combo.currentIndexChanged.connect(self._apply_default)
        default_row.addWidget(self.default_combo)
        default_row.addStretch(1)
        layout.addLayout(default_row)
        table = QtWidgets.QTableWidget(len(items), 2)
        table.setHorizontalHeaderLabels(["样本", "原始等级"])
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        table.setColumnWidth(1, 130)
        for row, (key, path) in enumerate(items):
            item = QtWidgets.QTableWidgetItem(key)
            item.setToolTip(path)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            table.setItem(row, 0, item)
            combo = self._new_combo()
            table.setCellWidget(row, 1, combo)
            self._combos[path] = combo
        layout.addWidget(table, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("保存等级")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._accept_assignments)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _new_combo(self) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        for text, value in self.OPTIONS:
            combo.addItem(text, value)
        return combo

    def _apply_default(self) -> None:
        index = self.default_combo.currentIndex()
        if index <= 0:
            return
        for combo in self._combos.values():
            combo.setCurrentIndex(index)

    def _accept_assignments(self) -> None:
        missing = [
            path for path, combo in self._combos.items()
            if combo.currentIndex() == 0
        ]
        if missing:
            QtWidgets.QMessageBox.information(
                self,
                "仍有样本未标定",
                "请为每个样本选择 A、B、C、X 或“未提供”。",
            )
            return
        self.assignments = {
            path: (str(combo.currentData()) or None)
            for path, combo in self._combos.items()
        }
        self.accept()


class LensAgentWindow(QtWidgets.QMainWindow):
    def __init__(self, project_root: str | Path):
        super().__init__()
        self.project = ProjectIndex(project_root)
        self.snapshots: dict[str, ResultSnapshot] = {}
        self.job_queue: deque[JobSpec] = deque()
        self.job_rows: dict[str, int] = {}
        self.process: QtCore.QProcess | None = None
        self.active_job: JobSpec | None = None
        self.review_process: QtCore.QProcess | None = None
        self.review_job: JobSpec | None = None
        self.review_progress: QtWidgets.QProgressDialog | None = None
        self.log_tail: list[str] = []
        self.review_log_tail: list[str] = []
        self.selected_object_id: str | None = None
        self.setWindowTitle("LensAgent — 强引力透镜候选体智能评审")
        self.resize(1680, 980)
        self.setMinimumSize(1180, 720)
        self._build_ui()
        self._setup_tray()
        self.apply_theme(self.project.theme)
        self.refresh_all()
        QtCore.QTimer.singleShot(250, self._prompt_unconfirmed_reference_grades)

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        shell = QtWidgets.QVBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        header = QtWidgets.QFrame()
        header.setObjectName("topbar")
        header.setFixedHeight(58)
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(14, 8, 16, 8)
        header_layout.setSpacing(9)
        header_layout.addWidget(BrandMark())
        brand = QtWidgets.QLabel("SL agent")
        brand.setObjectName("brand")
        header_layout.addWidget(brand)
        header_layout.addSpacing(12)
        self.project_label = QtWidgets.QLabel()
        self.project_label.setObjectName("topMeta")
        self.project_label.setMaximumWidth(180)
        self.sample_count_label = QtWidgets.QLabel("样本数：0")
        self.sample_count_label.setObjectName("topMeta")
        self.sample_count_label.setMaximumWidth(100)
        self.current_object_label = QtWidgets.QLabel("当前：—")
        self.current_object_label.setObjectName("topMeta")
        self.current_object_label.setMaximumWidth(230)
        for metadata in (
            self.project_label,
            self.sample_count_label,
            self.current_object_label,
        ):
            header_layout.addWidget(metadata)
            divider = QtWidgets.QFrame()
            divider.setObjectName("topDivider")
            divider.setFrameShape(QtWidgets.QFrame.Shape.VLine)
            header_layout.addWidget(divider)
        self.theme_button = QtWidgets.QPushButton()
        self.theme_button.setObjectName("themeToggle")
        self.theme_button.setFixedHeight(30)
        self.theme_button.setToolTip("切换日间/夜间模式")
        header_layout.addWidget(self.theme_button)
        header_layout.addStretch(1)

        self.import_files_button = QtWidgets.QPushButton("导入 FITS")
        self.import_folder_button = QtWidgets.QPushButton("导入文件夹")
        self.start_button = QtWidgets.QPushButton("开始跑批")
        self.import_files_button.setObjectName("topAction")
        self.import_folder_button.setObjectName("topAction")
        self.start_button.setObjectName("topPrimary")
        self.import_files_button.setIcon(_folder_icon())
        self.import_folder_button.setIcon(_folder_icon())
        self.start_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.import_files_button.setFixedSize(98, 36)
        self.import_folder_button.setFixedSize(110, 36)
        self.start_button.setFixedSize(96, 36)
        self.judgment_selector = QtWidgets.QComboBox()
        self.judgment_selector.addItem("固定规则", "rules")
        self.judgment_selector.addItem("规则 + DeepSeek", "deepseek")
        self.judgment_selector.setCurrentIndex(
            1 if LLMConfig.from_environment().backend == "deepseek" else 0)
        self.judgment_selector.setToolTip("DeepSeek 接收观测数值并产生实验性判断；通常调用两次 API，引用错误时最多纠正一次。")
        header_layout.addWidget(self.judgment_selector)
        header_layout.addWidget(self.import_files_button)
        header_layout.addWidget(self.import_folder_button)
        header_layout.addWidget(self.start_button)
        header_layout.addSpacing(10)
        self.header_task_dot = QtWidgets.QLabel("●")
        self.header_task_dot.setObjectName("activityDot")
        self.header_task_status = QtWidgets.QLabel("后台任务：空闲")
        self.header_task_status.setObjectName("topMeta")
        self.header_task_status.setMinimumWidth(112)
        header_layout.addWidget(self.header_task_dot)
        header_layout.addWidget(self.header_task_status)
        shell.addWidget(header)

        self.import_files_button.clicked.connect(self.import_files)
        self.import_folder_button.clicked.connect(self.import_folder)
        self.start_button.clicked.connect(self.start_batch)
        self.theme_button.clicked.connect(self._theme_changed)
        self._update_header_metadata()

        vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        vertical_splitter.setObjectName("workspaceSplitter")
        shell.addWidget(vertical_splitter, 1)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")
        vertical_splitter.addWidget(splitter)

        left = QtWidgets.QFrame()
        left.setObjectName("sidebar")
        left.setMinimumWidth(330)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(7, 8, 5, 5)
        left_layout.setSpacing(4)
        left_header_row = QtWidgets.QHBoxLayout()
        left_header = QtWidgets.QLabel("样本与结果")
        left_header.setObjectName("sectionTitle")
        left_header_row.addWidget(left_header)
        left_header_row.addStretch(1)
        self.object_count = QtWidgets.QLabel("0")
        self.object_count.setObjectName("countBadge")
        left_header_row.addWidget(self.object_count)
        self.search_box = QtWidgets.QLineEdit()
        self.search_box.setPlaceholderText("搜索样本…")
        self.search_box.setClearButtonEnabled(True)
        self.only_review = QtWidgets.QCheckBox("只看需复核")
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self.tree.header().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Fixed
        )
        self.tree.setColumnWidth(1, 52)
        self.tree.setIndentation(11)
        self.tree.setAnimated(True)
        self.tree.setExpandsOnDoubleClick(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIconSize(QtCore.QSize(14, 14))
        left_layout.addLayout(left_header_row)
        left_layout.addWidget(self.search_box)
        left_layout.addWidget(self.only_review)
        left_layout.addWidget(self.tree, 1)
        splitter.addWidget(left)

        middle = QtWidgets.QFrame()
        middle.setObjectName("viewerPanel")
        middle_layout = QtWidgets.QVBoxLayout(middle)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)
        viewer_header = QtWidgets.QFrame()
        viewer_header.setObjectName("viewerHeader")
        viewer_header_layout = QtWidgets.QHBoxLayout(viewer_header)
        viewer_header_layout.setContentsMargins(14, 9, 12, 9)
        self.object_title = QtWidgets.QLabel("请选择一个样本")
        self.object_title.setObjectName("viewerTitle")
        self.media_title = QtWidgets.QLabel("")
        self.media_title.setObjectName("muted")
        viewer_header_layout.addWidget(self.object_title)
        viewer_header_layout.addWidget(self.media_title)
        viewer_header_layout.addStretch(1)
        self.fit_button = QtWidgets.QPushButton("适应窗口")
        self.fit_button.setObjectName("toolButton")
        self.native_button = QtWidgets.QPushButton("1:1")
        self.native_button.setObjectName("toolButton")
        self.zoom_out_button = QtWidgets.QPushButton("−")
        self.zoom_out_button.setObjectName("toolButton")
        self.zoom_out_button.setFixedWidth(32)
        self.zoom_out_button.setToolTip("缩小图片（也可在图片上向下滚动鼠标滚轮）")
        self.zoom_label = QtWidgets.QLabel("100%")
        self.zoom_label.setObjectName("muted")
        self.zoom_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.zoom_label.setFixedWidth(52)
        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_in_button.setObjectName("toolButton")
        self.zoom_in_button.setFixedWidth(32)
        self.zoom_in_button.setToolTip("放大图片（也可在图片上向上滚动鼠标滚轮）")
        self.open_button = QtWidgets.QPushButton("新窗口")
        self.open_button.setObjectName("toolButton")
        viewer_header_layout.addWidget(self.fit_button)
        viewer_header_layout.addWidget(self.native_button)
        viewer_header_layout.addWidget(self.zoom_out_button)
        viewer_header_layout.addWidget(self.zoom_label)
        viewer_header_layout.addWidget(self.zoom_in_button)
        viewer_header_layout.addWidget(self.open_button)
        middle_layout.addWidget(viewer_header)
        self.viewer_stack = QtWidgets.QStackedWidget()
        self.empty_view = QtWidgets.QLabel("从左侧展开一个样本\n选择过程图或代码结果")
        self.empty_view.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.empty_view.setObjectName("emptyViewer")
        self.image_area = QtWidgets.QScrollArea()
        self.image_area.setWidgetResizable(True)
        self.image_area.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.image_canvas = ImageCanvas()
        self.image_area.setWidget(self.image_canvas)
        self.text_view = QtWidgets.QPlainTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.text_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_view.setObjectName("codeViewer")
        self.report_view = QtWidgets.QTextBrowser()
        self.report_view.setObjectName("reportViewer")
        self.report_view.setOpenExternalLinks(False)
        self.report_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.viewer_stack.addWidget(self.empty_view)
        self.viewer_stack.addWidget(self.image_area)
        self.viewer_stack.addWidget(self.text_view)
        self.viewer_stack.addWidget(self.report_view)
        middle_layout.addWidget(self.viewer_stack, 1)
        self.current_media_path = ""
        self.fit_button.clicked.connect(lambda: self._set_image_mode(True))
        self.native_button.clicked.connect(lambda: self._set_image_mode(False))
        self.zoom_out_button.clicked.connect(lambda: self._zoom_image(1 / 1.25))
        self.zoom_in_button.clicked.connect(lambda: self._zoom_image(1.25))
        self.image_canvas.zoomRequested.connect(self._zoom_image)
        self.image_canvas.scaleChanged.connect(self._update_zoom_label)
        self.open_button.clicked.connect(self._open_current_media)
        splitter.addWidget(middle)

        right = QtWidgets.QFrame()
        right.setObjectName("inspector")
        right.setMinimumWidth(330)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)
        inspector_title = QtWidgets.QLabel("评审结论")
        inspector_title.setObjectName("sectionTitle")
        right_layout.addWidget(inspector_title)
        self.detail_tabs = QtWidgets.QTabWidget()
        self.assessment_area = QtWidgets.QScrollArea()
        self.assessment_area.setWidgetResizable(True)
        self.assessment_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.assessment_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history_list = QtWidgets.QListWidget()
        self.detail_tabs.addTab(self.assessment_area, "评审结果")
        self.detail_tabs.addTab(self.history_list, "历史记录")
        right_layout.addWidget(self.detail_tabs, 1)
        action_bar = QtWidgets.QFrame()
        action_bar.setObjectName("inspectorActions")
        action_layout = QtWidgets.QHBoxLayout(action_bar)
        action_layout.setContentsMargins(0, 8, 0, 0)
        self.correct_button = QtWidgets.QPushButton("人工校正")
        self.correct_button.setObjectName("correct")
        self.correct_button.clicked.connect(self.start_manual_review)
        self.report_button = QtWidgets.QPushButton("查看完整报告")
        self.report_button.setObjectName("primary")
        self.report_button.clicked.connect(self.open_report)
        action_layout.addWidget(self.correct_button)
        action_layout.addWidget(self.report_button)
        right_layout.addWidget(action_bar)
        splitter.addWidget(right)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, False)
        splitter.setSizes([350, 950, 350])

        queue_panel = QtWidgets.QFrame()
        queue_panel.setObjectName("activityPanel")
        queue_layout = QtWidgets.QVBoxLayout(queue_panel)
        queue_layout.setContentsMargins(10, 7, 10, 7)
        queue_layout.setSpacing(5)
        queue_header = QtWidgets.QHBoxLayout()
        queue_title = QtWidgets.QLabel("后台任务")
        queue_title.setObjectName("activityTitle")
        queue_header.addWidget(queue_title)
        queue_header.addStretch(1)
        self.queue_status = QtWidgets.QLabel("空闲")
        self.queue_status.setObjectName("muted")
        queue_header.addWidget(self.queue_status)
        self.queue_table = QtWidgets.QTableWidget(0, 4)
        self.queue_table.setHorizontalHeaderLabels(["样本", "任务", "状态", "输出目录"])
        self.queue_table.horizontalHeader().setStretchLastSection(True)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setShowGrid(False)
        self.queue_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        queue_layout.addLayout(queue_header)
        queue_layout.addWidget(self.queue_table)
        vertical_splitter.addWidget(queue_panel)
        vertical_splitter.setStretchFactor(0, 1)
        vertical_splitter.setStretchFactor(1, 0)
        vertical_splitter.setSizes([820, 115])
        self.workspace_splitter = vertical_splitter

        self.search_box.textChanged.connect(self.refresh_tree)
        self.only_review.toggled.connect(self.refresh_tree)
        self.tree.itemSelectionChanged.connect(self._tree_selection_changed)
        self.statusBar().showMessage("就绪")

    def _setup_tray(self) -> None:
        self.tray = QtWidgets.QSystemTrayIcon(self)
        self.tray.setIcon(self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_ComputerIcon))
        self.tray.setToolTip("LensAgent")
        self.tray.show()

    def apply_theme(self, mode: str) -> None:
        self.project.theme = mode
        self.project.save()
        self.setStyleSheet(stylesheet(mode))
        if hasattr(self, "theme_button"):
            self.theme_button.setText("☾ 夜间" if mode == "dark" else "☀ 日间")

    def _theme_changed(self) -> None:
        self.apply_theme("light" if self.project.theme == "dark" else "dark")

    def _update_header_metadata(self) -> None:
        project_name = self.project.root.name or str(self.project.root)
        self.project_label.setText(f"项目：{project_name}")
        self.project_label.setToolTip(str(self.project.root))
        self.sample_count_label.setText(f"样本数：{len(self.project.objects):,}")
        current = self.selected_object_id or "—"
        self.current_object_label.setText(f"当前：{current}")
        self.current_object_label.setToolTip(current)
        self.theme_button.setText("☾ 夜间" if self.project.theme == "dark" else "☀ 日间")

    def change_project(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择 LensAgent 项目输出目录", str(self.project.root)
        )
        if not selected:
            return
        self.project = ProjectIndex(selected)
        self.snapshots.clear()
        self.job_queue.clear()
        self.selected_object_id = None
        self.apply_theme(self.project.theme)
        self._update_header_metadata()
        self.refresh_all()
        QtCore.QTimer.singleShot(0, self._prompt_unconfirmed_reference_grades)

    def import_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "导入 FITS", "", "FITS (*.fits *.fit *.fts)"
        )
        self._add_inputs([Path(item) for item in paths])

    def import_folder(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "导入包含 FITS 的文件夹")
        if not selected:
            return
        found = discover_inputs(selected)
        if not found:
            QtWidgets.QMessageBox.information(self, "没有 FITS", "所选文件夹中没有找到 FITS 文件。")
            return
        self._add_inputs(found)

    def _add_inputs(self, paths: list[Path]) -> None:
        paths = [Path(item).expanduser().resolve() for item in paths]
        new_paths = []
        for path in paths:
            object_id = object_id_from_input(path)
            existing = self.project.objects.get(object_id)
            if existing is None or Path(existing.input_path) != path:
                new_paths.append(path)
        unresolved = [
            path for path in new_paths if infer_reference_grade(path)[0] is None
        ]
        assignments: dict[str, str | None] = {}
        if unresolved:
            dialog = ReferenceGradeDialog(
                [(object_id_from_input(path), str(path)) for path in unresolved], self
            )
            if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                self.statusBar().showMessage("已取消导入；没有添加新样本。", 5000)
                return
            assignments = dialog.assignments
        added = self.project.add_inputs(paths)
        for record in added:
            key = str(Path(record.input_path).expanduser().resolve())
            if key not in assignments:
                continue
            record.reference_grade = assignments[key]
            record.reference_grade_source = (
                "人工标定" if assignments[key] else "人工确认：未提供"
            )
            record.reference_grade_confirmed = True
        self.project.save()
        self.statusBar().showMessage(f"已加入 {len(added)} 个样本；点击“开始跑批”运行。", 6000)
        self._update_header_metadata()
        self.refresh_tree()

    def _prompt_unconfirmed_reference_grades(self) -> None:
        records = [
            record for record in self.project.objects.values()
            if not record.reference_grade_confirmed
        ]
        if not records:
            return
        dialog = ReferenceGradeDialog(
            [(record.object_id, record.input_path) for record in records], self
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        for record in records:
            grade = dialog.assignments.get(record.input_path)
            record.reference_grade = grade
            record.reference_grade_source = (
                "人工标定" if grade else "人工确认：未提供"
            )
            record.reference_grade_confirmed = True
        self.project.save()
        self.refresh_all()

    def start_batch(self) -> None:
        queued = 0
        for record in self.project.objects.values():
            if record.runs or record.status not in {"queued", "interrupted"}:
                continue
            run = self.project.new_run(record, kind="auto")
            self._enqueue_analyze(record, run)
            queued += 1
        if queued == 0 and not self.job_queue and self.process is None:
            self.statusBar().showMessage("没有新的等待样本。", 4000)
            return
        self.refresh_all()
        self._start_next_job()

    def _enqueue_analyze(self, record: ObjectRecord, run: RunRecord) -> None:
        self.job_queue.append(
            JobSpec(
                object_id=record.object_id,
                run_id=run.run_id,
                program=sys.executable,
                arguments=[
                    "-m", "expert_agent", "analyze", record.input_path,
                    "--output", run.output_root,
                    "--judgment", str(self.judgment_selector.currentData()),
                ],
                label="自动分析" if run.kind == "auto" else "人工重跑",
            )
        )

    def _find_run(self, object_id: str, run_id: str) -> tuple[ObjectRecord, RunRecord]:
        record = self.project.objects[object_id]
        run = next(item for item in record.runs if item.run_id == run_id)
        return record, run

    def _start_next_job(self) -> None:
        if self.process is not None or not self.job_queue:
            self._update_queue_status()
            return
        job = self.job_queue.popleft()
        record, run = self._find_run(job.object_id, job.run_id)
        run.status = "running"
        run.started_utc = utc_now()
        record.status = "running"
        self.project.save()
        self.active_job = job
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_process_output)
        process.finished.connect(self._process_finished)
        process.errorOccurred.connect(self._process_error)
        self.process = process
        self.log_tail.clear()
        process.start(job.program, job.arguments)
        self.refresh_all()

    def _read_process_output(self) -> None:
        if self.process is None:
            return
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log_tail.extend(chunk.splitlines())
        self.log_tail = self.log_tail[-60:]
        if chunk.strip():
            self.statusBar().showMessage(chunk.strip().splitlines()[-1][:180])

    def _process_error(self, error) -> None:
        self.statusBar().showMessage(f"后台任务错误：{error}", 8000)

    def _process_finished(self, exit_code: int, exit_status) -> None:
        job = self.active_job
        if job is None:
            return
        record, run = self._find_run(job.object_id, job.run_id)
        run.finished_utc = utc_now()
        if exit_code in {0, 3}:
            try:
                snapshot = load_result_snapshot(run.output_dir)
                self.snapshots[record.object_id] = snapshot
                run.status = "completed"
                run.grade = snapshot.grade
                record.status = "completed"
                if exit_code == 3:
                    run.error = "DeepSeek 未完成；规则结果可用，详情见模型评审状态。"
                self._notify("LensAgent 分析完成", f"{record.object_id}：{snapshot.grade}")
            except Exception as exc:
                run.status = "failed"
                run.error = str(exc)
                record.status = "failed"
        else:
            run.status = "failed"
            run.error = "\n".join(self.log_tail[-12:]) or f"exit code {exit_code}"
            record.status = "failed"
            self._notify("LensAgent 任务失败", record.object_id)
        self.project.save()
        self.process.deleteLater() if self.process is not None else None
        self.process = None
        self.active_job = None
        self.refresh_all()
        self._start_next_job()

    def start_manual_review(self) -> None:
        if not self.selected_object_id:
            return
        if self.review_process is not None:
            QtWidgets.QMessageBox.information(self, "人工校正", "已有一个人工校正窗口正在运行。")
            return
        record = self.project.objects[self.selected_object_id]
        current = record.current_run()
        if current is None or current.status != "completed":
            QtWidgets.QMessageBox.information(self, "人工校正", "该样本还没有可供校正的完整结果。")
            return
        run = self.project.new_run(record, kind="human", parent_run_id=current.run_id)
        record.status = "reviewing"
        run.status = "running"
        run.started_utc = utc_now()
        self.project.save()
        job = JobSpec(
            object_id=record.object_id,
            run_id=run.run_id,
            program=sys.executable,
            arguments=[
                "-m", "expert_agent", "foreground-review", current.output_dir,
                "--output", run.output_root,
                "--ui-theme", self.project.theme,
                "--judgment", str(self.judgment_selector.currentData()),
            ],
            label="人工校正与重跑",
        )
        process = QtCore.QProcess(self)
        process.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        process.readyReadStandardOutput.connect(self._read_review_output)
        process.finished.connect(self._review_finished)
        process.errorOccurred.connect(self._review_process_error)
        self.review_process = process
        self.review_job = job
        self.review_log_tail.clear()
        self.correct_button.setText("人工界面准备中…")
        self.correct_button.setEnabled(False)
        self.review_progress = QtWidgets.QProgressDialog(
            "正在读取已有分析结果并创建人工检验界面…", "", 0, 0, self
        )
        self.review_progress.setWindowTitle("人工校正")
        self.review_progress.setCancelButton(None)
        self.review_progress.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self.review_progress.setMinimumDuration(0)
        self.review_progress.setAutoClose(False)
        self.review_progress.show()
        process.start(job.program, job.arguments)
        self.statusBar().showMessage("正在读取已有分析结果并打开人工校正界面…", 0)
        self.refresh_all()

    def _read_review_output(self) -> None:
        if self.review_process is None:
            return
        chunk = bytes(self.review_process.readAllStandardOutput()).decode(
            "utf-8", errors="replace"
        )
        self.review_log_tail.extend(chunk.splitlines())
        self.review_log_tail = self.review_log_tail[-80:]
        if "opening editor" in chunk:
            if self.review_progress is not None:
                self.review_progress.setLabelText("已有结果读取完成，正在创建人工检验窗口…")
        if "MANUAL_REVIEW_WINDOW_READY" in chunk:
            if self.review_progress is not None:
                self.review_progress.close()
                self.review_progress.deleteLater()
                self.review_progress = None
            self.correct_button.setText("人工校正进行中")
            self.statusBar().showMessage(
                "人工校正窗口已打开；完成标注后将在后台完整重跑。", 0
            )

    def _review_process_error(self, error) -> None:
        if self.review_progress is not None:
            self.review_progress.close()
            self.review_progress.deleteLater()
            self.review_progress = None
        message = f"人工校正进程无法启动：{error}"
        self.statusBar().showMessage(message, 10000)
        QtWidgets.QMessageBox.critical(self, "人工校正启动失败", message)

    def _review_finished(self, exit_code: int, exit_status) -> None:
        job = self.review_job
        if job is None:
            return
        record, run = self._find_run(job.object_id, job.run_id)
        run.finished_utc = utc_now()
        if exit_code in {0, 3}:
            try:
                snapshot = load_result_snapshot(run.output_dir)
                self.snapshots[record.object_id] = snapshot
                run.status = "completed"
                run.grade = snapshot.grade
                record.current_run_id = run.run_id
                record.status = "completed"
                if exit_code == 3:
                    run.error = "DeepSeek 未完成；规则结果可用，详情见模型评审状态。"
                self._notify("人工重跑完成", f"{record.object_id}：{snapshot.grade}")
            except Exception as exc:
                run.status = "failed"
                run.error = str(exc)
                record.status = "failed"
        else:
            details = "\n".join(self.review_log_tail[-16:]).strip()
            cancelled = "closed without accepting an annotation" in details
            run.status = "cancelled" if cancelled else "failed"
            run.error = details or "标注窗口被取消或重跑失败"
            record.status = "completed" if any(item.status == "completed" for item in record.runs) else "failed"
            completed = [item for item in record.runs if item.status == "completed"]
            if completed:
                record.current_run_id = completed[-1].run_id
            if not cancelled:
                QtWidgets.QMessageBox.critical(
                    self,
                    "人工校正失败",
                    "人工界面未能完成启动或重跑。\n\n" + (details[-1800:] or f"退出码：{exit_code}"),
                )
        self.project.save()
        self.review_process.deleteLater() if self.review_process is not None else None
        self.review_process = None
        self.review_job = None
        if self.review_progress is not None:
            self.review_progress.close()
            self.review_progress.deleteLater()
            self.review_progress = None
        self.correct_button.setText("人工校正")
        self.statusBar().clearMessage()
        self.refresh_all()

    def _notify(self, title: str, message: str) -> None:
        self.tray.showMessage(title, message, self.tray.icon(), 6000)
        self.statusBar().showMessage(f"{title}：{message}", 8000)

    def _load_snapshot(self, record: ObjectRecord) -> ResultSnapshot | None:
        cached = self.snapshots.get(record.object_id)
        run = record.current_run()
        if cached is not None and run is not None and cached.output_dir == run.output_dir:
            return cached
        if run is None or run.status != "completed":
            return None
        try:
            snapshot = load_result_snapshot(run.output_dir)
        except Exception:
            return None
        self.snapshots[record.object_id] = snapshot
        return snapshot

    def refresh_all(self) -> None:
        for object_id in self.project.sync_completed_runs():
            self.snapshots.pop(object_id, None)
        self.refresh_tree()
        self.refresh_queue()
        if self.selected_object_id in self.project.objects:
            self.show_object(self.selected_object_id)
        else:
            self._show_empty()

    def refresh_tree(self) -> None:
        selected = self.selected_object_id
        query = self.search_box.text().strip().lower() if hasattr(self, "search_box") else ""
        only_review = self.only_review.isChecked() if hasattr(self, "only_review") else False
        expanded_objects: set[str] = set()
        for index in range(self.tree.topLevelItemCount()):
            group_item = self.tree.topLevelItem(index)
            for child_index in range(group_item.childCount()):
                child = group_item.child(child_index)
                if child.data(0, ROLE_KIND) == "object" and child.isExpanded():
                    expanded_objects.add(str(child.data(0, ROLE_OBJECT)))
        self.tree.blockSignals(True)
        self.tree.clear()
        groups = {name: QtWidgets.QTreeWidgetItem([name, ""]) for name in GROUP_ORDER}
        counts = {name: 0 for name in GROUP_ORDER}
        visible_count = 0
        for record in self.project.objects.values():
            if query and query not in record.object_id.lower():
                continue
            snapshot = self._load_snapshot(record)
            if only_review and not (snapshot and snapshot.needs_review):
                continue
            if record.status in {"running", "reviewing"}:
                group = "运行中"
            elif record.status in {"failed", "cancelled", "interrupted"}:
                group = "运行失败"
            elif not record.runs or record.status == "queued":
                group = "等待运行"
            else:
                group = snapshot.group if snapshot else "无法判断"
            reference_text = record.reference_grade or "—"
            item = QtWidgets.QTreeWidgetItem([record.object_id, reference_text])
            item.setData(0, ROLE_KIND, "object")
            item.setData(0, ROLE_OBJECT, record.object_id)
            item.setToolTip(0, record.input_path)
            item.setTextAlignment(1, QtCore.Qt.AlignmentFlag.AlignCenter)
            item.setForeground(1, QtGui.QBrush(QtGui.QColor(PALETTE["blue"])))
            item.setToolTip(
                1,
                (
                    f"Euclid 原始等级：{record.reference_grade}\n"
                    f"{record.reference_grade_source}\n"
                    "仅用于结果对照，不参与 Agent 评级。"
                    if record.reference_grade
                    else "Euclid 原始等级：未提供\n仅用于结果对照，不参与 Agent 评级。"
                ),
            )
            if snapshot and snapshot.needs_review:
                item.setForeground(0, QtGui.QBrush(QtGui.QColor(PALETTE["pink"])))
                item.setToolTip(0, f"建议人工复核\n{record.input_path}")
            if snapshot:
                images = QtWidgets.QTreeWidgetItem(["过程图"])
                images.setData(0, ROLE_KIND, "section")
                images.setForeground(0, QtGui.QBrush(QtGui.QColor(PALETTE["blue"])))
                for title, path in self._artifact_items(record, snapshot):
                    leaf = QtWidgets.QTreeWidgetItem([title])
                    leaf.setData(0, ROLE_KIND, "image")
                    leaf.setData(0, ROLE_OBJECT, record.object_id)
                    leaf.setData(0, ROLE_PATH, path)
                    leaf.setData(0, ROLE_TITLE, title)
                    leaf.setToolTip(0, path)
                    images.addChild(leaf)
                item.addChild(images)
                results = QtWidgets.QTreeWidgetItem(["代码与报告"])
                results.setData(0, ROLE_KIND, "section")
                results.setForeground(0, QtGui.QBrush(QtGui.QColor(PALETTE["pink"])))
                run = record.current_run()
                if run:
                    for filename, _title in RESULT_FILES:
                        path = Path(run.output_dir) / filename
                        if not path.exists():
                            continue
                        leaf = QtWidgets.QTreeWidgetItem([filename])
                        leaf.setData(0, ROLE_KIND, "text")
                        leaf.setData(0, ROLE_OBJECT, record.object_id)
                        leaf.setData(0, ROLE_PATH, str(path))
                        leaf.setData(0, ROLE_TITLE, filename)
                        leaf.setToolTip(0, str(path))
                        results.addChild(leaf)
                item.addChild(results)
            groups[group].addChild(item)
            counts[group] += 1
            visible_count += 1
            item.setExpanded(record.object_id in expanded_objects)
            if record.object_id == selected:
                self.tree.setCurrentItem(item)
        colors = {
            "A": "#fdc0e1",
            "B": "#fecee6",
            "C": "#ffeef8",
            "X": "#A1CDC4",
            "无法判断": "#8ABAB1",
            "运行中": "#9ecafe",
            "等待运行": "#71b0fd",
            "运行失败": "#8b98a5",
        }
        for name in GROUP_ORDER:
            group_item = groups[name]
            group_item.setText(0, f"{name} ({counts[name]})")
            group_item.setIcon(0, _circle_icon(colors[name]))
            self.tree.addTopLevelItem(group_item)
            group_item.setExpanded(counts[name] > 0)
            group_item.setData(0, ROLE_KIND, "group")
        self.object_count.setText(str(visible_count))
        self._update_header_metadata()
        self.tree.blockSignals(False)

    def _tree_selection_changed(self) -> None:
        items = self.tree.selectedItems()
        if not items:
            return
        item = items[0]
        kind = item.data(0, ROLE_KIND)
        object_id = item.data(0, ROLE_OBJECT)
        if not object_id:
            return
        self.selected_object_id = str(object_id)
        self._update_header_metadata()
        if kind == "image":
            self.show_object(self.selected_object_id, select_default=False)
            self._show_image(str(item.data(0, ROLE_TITLE)), str(item.data(0, ROLE_PATH)))
        elif kind == "text":
            self.show_object(self.selected_object_id, select_default=False)
            self._show_text(str(item.data(0, ROLE_TITLE)), str(item.data(0, ROLE_PATH)))
        elif kind == "object":
            self.show_object(self.selected_object_id, select_default=True)

    def _artifact_items(
        self, record: ObjectRecord, snapshot: ResultSnapshot
    ) -> list[tuple[str, str]]:
        paths = dict(snapshot.artifacts)
        rgb = Path(snapshot.input_path).with_name("rgb_0.png")
        if rgb.exists():
            paths["input_rgb"] = str(rgb)
        try:
            paths["input_vis"] = str(
                render_input_preview(snapshot.input_path, self.project.root / ".ui_cache")
            )
        except Exception:
            pass
        ordered = [
            (Path(paths[key]).name, paths[key])
            for key, _title in CORE_ARTIFACTS
            if key in paths and Path(paths[key]).exists()
        ]
        used = {path for _, path in ordered}
        ordered.extend(
            (Path(path).name, path)
            for _key, path in paths.items()
            if Path(path).suffix.lower() in {".png", ".jpg", ".jpeg"}
            and Path(path).exists()
            and path not in used
        )
        return ordered

    def show_object(self, object_id: str, *, select_default: bool = True) -> None:
        record = self.project.objects[object_id]
        snapshot = self._load_snapshot(record)
        self.object_title.setText(object_id)
        self._populate_history(record)
        if snapshot is None:
            self.viewer_stack.setCurrentWidget(self.empty_view)
            self.media_title.setText("等待分析")
            self.current_media_path = ""
            self._set_assessment(record, None)
            return
        artifacts = self._artifact_items(record, snapshot)
        if select_default and artifacts:
            self._show_image(*artifacts[0])
        elif select_default:
            self.viewer_stack.setCurrentWidget(self.empty_view)
            self.media_title.setText("没有可显示的过程图")
        self._set_assessment(record, snapshot)

    def _show_image(self, title: str, path: str) -> None:
        self.media_title.setText(f"/  {title}")
        self.current_media_path = path
        self.image_area.setWidgetResizable(True)
        self.image_canvas.fit_mode = True
        self.image_canvas.set_source(path)
        self.viewer_stack.setCurrentWidget(self.image_area)
        self._set_zoom_controls_enabled(True)
        self.open_button.setEnabled(True)

    def _show_text(self, title: str, path: str) -> None:
        self.media_title.setText(f"/  {title}")
        self.current_media_path = path
        try:
            content = Path(path).read_text(encoding="utf-8")
        except Exception as exc:
            content = f"无法读取文件：{exc}"
        if Path(path).suffix.lower() == ".md":
            base = QtCore.QUrl.fromLocalFile(str(Path(path).parent) + "/")
            self.report_view.document().setBaseUrl(base)
            self.report_view.setMarkdown(content)
            self.report_view.verticalScrollBar().setValue(0)
            self.viewer_stack.setCurrentWidget(self.report_view)
        else:
            self.text_view.setPlainText(content)
            self.text_view.verticalScrollBar().setValue(0)
            self.viewer_stack.setCurrentWidget(self.text_view)
        self._set_zoom_controls_enabled(False)
        self.open_button.setEnabled(True)

    def _set_image_mode(self, fit: bool) -> None:
        self.image_canvas.fit_mode = fit
        self.image_area.setWidgetResizable(fit)
        if not fit:
            self.image_canvas.zoom_factor = 1.0
        self.image_canvas.updateGeometry()
        self.image_canvas.refresh_scale()

    def _zoom_image(self, multiplier: float) -> None:
        if self.image_canvas.source_pixmap.isNull():
            return
        horizontal = self.image_area.horizontalScrollBar()
        vertical = self.image_area.verticalScrollBar()
        old_h_ratio = horizontal.value() / max(1, horizontal.maximum())
        old_v_ratio = vertical.value() / max(1, vertical.maximum())
        self.image_area.setWidgetResizable(False)
        self.image_canvas.set_zoom(self.image_canvas.zoom_factor * multiplier)
        horizontal.setValue(round(old_h_ratio * horizontal.maximum()))
        vertical.setValue(round(old_v_ratio * vertical.maximum()))

    def _update_zoom_label(self, factor: float) -> None:
        self.zoom_label.setText(f"{factor * 100:.0f}%")

    def _set_zoom_controls_enabled(self, enabled: bool) -> None:
        self.fit_button.setEnabled(enabled)
        self.native_button.setEnabled(enabled)
        self.zoom_out_button.setEnabled(enabled)
        self.zoom_in_button.setEnabled(enabled)
        self.zoom_label.setEnabled(enabled)

    def _open_current_media(self) -> None:
        if not self.current_media_path:
            return
        path = Path(self.current_media_path)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            self.open_image(str(path))
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def open_image(self, path: str) -> None:
        dialog = ImagePreview(path, self)
        dialog.exec()

    def _set_assessment(self, record: ObjectRecord, snapshot: ResultSnapshot | None) -> None:
        widget = QtWidgets.QWidget()
        widget.setMinimumWidth(0)
        widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(12)
        if snapshot is None:
            status = QtWidgets.QLabel(f"状态：{record.status}")
            status.setObjectName("sectionTitle")
            layout.addWidget(status)
            layout.addWidget(QtWidgets.QLabel("分析完成后将在这里显示判定依据。"))
            layout.addStretch(1)
            self.correct_button.setEnabled(False)
            self.report_button.setEnabled(False)
            self.assessment_area.setWidget(widget)
            return
        self.correct_button.setEnabled(self.review_process is None)
        self.report_button.setEnabled(True)
        grade_card = QtWidgets.QFrame()
        grade_card.setObjectName("gradeCard")
        grade_card.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        grade_row = QtWidgets.QHBoxLayout(grade_card)
        grade_row.setContentsMargins(14, 12, 14, 12)
        grade_copy = QtWidgets.QVBoxLayout()
        grade_kicker = QtWidgets.QLabel("规则暂定等级")
        grade_kicker.setObjectName("gradeKicker")
        grade_copy.addWidget(grade_kicker)
        grade = QtWidgets.QLabel(snapshot.grade)
        grade.setObjectName("grade")
        grade_copy.addWidget(grade)
        grade_row.addLayout(grade_copy)
        grade_row.addStretch(1)
        reference_pill = QtWidgets.QLabel(
            f"原始等级\n{record.reference_grade or '—'}"
        )
        reference_pill.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        reference_pill.setObjectName("gradePill")
        reference_pill.setToolTip(
            (
                f"{record.reference_grade_source}\n仅用于结果对照，不参与 Agent 评级。"
                if record.reference_grade_source
                else "原始数据未提供可识别的等级标签。"
            )
        )
        grade_row.addWidget(reference_pill)
        layout.addWidget(grade_card)
        review = snapshot.llm_assessment
        if review is not None:
            if review.get("status") == "completed":
                vote = review["assessment"]
                label = "无法判断" if vote["vote_category"] == "insufficient_data" else vote["vote_category"]
                agreement = "与规则票一致" if review.get("agrees_with_rules") else "与规则票不同，建议人工复核"
                layout.addWidget(self._text_section(
                    f"DeepSeek 暂定等级：{label}", f"{agreement}\n{vote['rationale']}"))
                layout.addWidget(self._bullet_section(
                    "DeepSeek 测量依据", [item['statement'] for item in vote['supporting_evidence']], "support"))
                layout.addWidget(self._bullet_section(
                    "DeepSeek 限制及竞争解释", vote['limitations'] + vote['competing_explanations'], "limit"))
            else:
                layout.addWidget(self._text_section(
                    "DeepSeek 未完成", f"{failure_message(review.get('error', 'unknown'))}\n本次没有模型等级，规则结果仍可查看。"))
        if snapshot.needs_review:
            warning = QtWidgets.QLabel("⚠ 前景或数据质量建议人工复核")
            warning.setObjectName("reviewWarning")
            warning.setWordWrap(True)
            layout.addWidget(warning)
        layout.addWidget(self._text_section("判定摘要", _cn_rationale(snapshot.rationale)))
        layout.addWidget(self._bullet_section("支持依据", snapshot.supporting, "support"))
        layout.addWidget(self._bullet_section("限制因素", snapshot.limiting, "limit"))
        qa_title = QtWidgets.QLabel("运行审计")
        qa_title.setObjectName("subheading")
        layout.addWidget(qa_title)
        qa = QtWidgets.QLabel(
            f"前景拟合：{'通过' if snapshot.model_success else '未通过'}\n"
            f"数据可用：{'是' if snapshot.data_usable else '否'}\n"
            f"颜色证据：{COLOUR_STATE_CN.get(snapshot.colour_state, '未知')}\n"
            f"决策审计：{'通过' if snapshot.audit_passed else '未通过'}"
        )
        qa.setWordWrap(True)
        qa.setObjectName("auditBox")
        layout.addWidget(qa)
        layout.addStretch(1)
        self.assessment_area.setWidget(widget)

    def _text_section(self, title: str, value: str):
        section = QtWidgets.QFrame()
        section.setObjectName("sectionBlock")
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("subheading")
        layout.addWidget(heading)
        body = QtWidgets.QFrame()
        body.setObjectName("infoBox")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(10, 9, 10, 9)
        label = QtWidgets.QLabel(value)
        label.setWordWrap(True)
        label.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Ignored,
            QtWidgets.QSizePolicy.Policy.Preferred,
        )
        label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        body_layout.addWidget(label)
        layout.addWidget(body)
        return section

    def _bullet_section(self, title: str, values: list[str], variant: str):
        section = QtWidgets.QFrame()
        section.setObjectName("sectionBlock")
        layout = QtWidgets.QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        heading = QtWidgets.QLabel(title)
        heading.setObjectName("subheading")
        layout.addWidget(heading)
        body = QtWidgets.QFrame()
        body.setObjectName("infoBox")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(10, 9, 10, 9)
        body_layout.setSpacing(7)
        if not values:
            empty = QtWidgets.QLabel("无")
            empty.setObjectName("muted")
            body_layout.addWidget(empty)
        for value in values:
            label = QtWidgets.QLabel(f"•  {_cn_evidence(value)}")
            label.setWordWrap(True)
            label.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Ignored,
                QtWidgets.QSizePolicy.Policy.Preferred,
            )
            label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            body_layout.addWidget(label)
        layout.addWidget(body)
        return section

    def _populate_history(self, record: ObjectRecord) -> None:
        self.history_list.clear()
        for run in reversed(record.runs):
            grade = f" · {run.grade}" if run.grade else ""
            self.history_list.addItem(
                f"{run.run_id}\n{run.kind} · {run.status}{grade}\n{run.output_dir}"
            )

    def open_report(self) -> None:
        if not self.selected_object_id:
            return
        record = self.project.objects[self.selected_object_id]
        run = record.current_run()
        if run is None:
            return
        report = Path(run.output_dir) / "report.md"
        if report.exists():
            self._show_text("report.md", str(report))

    def refresh_queue(self) -> None:
        entries: list[tuple[str, str, str, str]] = []
        if self.active_job:
            _, run = self._find_run(self.active_job.object_id, self.active_job.run_id)
            entries.append((self.active_job.object_id, self.active_job.label, "运行中", run.output_dir))
        if self.review_job:
            _, run = self._find_run(self.review_job.object_id, self.review_job.run_id)
            entries.append((self.review_job.object_id, self.review_job.label, "人工校正/重跑中", run.output_dir))
        for job in self.job_queue:
            _, run = self._find_run(job.object_id, job.run_id)
            entries.append((job.object_id, job.label, "等待", run.output_dir))
        self.queue_table.setRowCount(len(entries))
        for row, values in enumerate(entries):
            for column, value in enumerate(values):
                self.queue_table.setItem(row, column, QtWidgets.QTableWidgetItem(value))
        self._update_queue_status()

    def _update_queue_status(self) -> None:
        running = int(self.process is not None) + int(self.review_process is not None)
        waiting = len(self.job_queue)
        self.queue_status.setText(f"运行中 {running} · 等待 {waiting}")
        if running:
            header_status = f"后台任务：{running} 个运行中"
        elif waiting:
            header_status = f"后台任务：{waiting} 个等待中"
        else:
            header_status = "后台任务：空闲"
        self.header_task_status.setText(header_status)

    def _show_empty(self) -> None:
        self.object_title.setText("请选择一个样本")
        self.media_title.setText("")
        self.current_media_path = ""
        self.viewer_stack.setCurrentWidget(self.empty_view)
        self._set_zoom_controls_enabled(False)
        self.open_button.setEnabled(False)
        self.correct_button.setEnabled(False)
        self.report_button.setEnabled(False)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        label = QtWidgets.QLabel("导入 FITS 或文件夹后，样本会出现在左侧列表。")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label.setObjectName("muted")
        layout.addWidget(label)
        self.assessment_area.setWidget(widget)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lens-agent-ui")
    parser.add_argument(
        "--project",
        default=str(Path.cwd() / "lensagent_ui_outputs"),
        help="UI project/output directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("LensAgent")
    window = LensAgentWindow(args.project)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
