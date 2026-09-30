from __future__ import annotations


PALETTE = {
    "accent": "#B7E0FA",
    "pink": "#fdc0e1",
    "pink_mid": "#fecee6",
    "pink_soft": "#ffeef8",
    "blue_soft": "#c4dffc",
    "blue": "#71b0fd",
    "blue_deep": "#337BAC",
    "mist": "#D5E8F1",
    "cyan_soft": "#CAEBE7",
    "cyan": "#4FB1B2",
    "green": "#A9D9BB",
}


def stylesheet(mode: str) -> str:
    dark = mode == "dark"
    bg = "#000000" if dark else "#ffffff"
    surface = "#080c12" if dark else "#fbfdff"
    surface_alt = "#0d141e" if dark else "#f3f8fc"
    surface_high = "#111c28" if dark else "#ffffff"
    text = "#f5f9fd" if dark else "#122231"
    muted = "#91a7ba" if dark else "#5a6f80"
    border = "#1e3448" if dark else "#d5e8f1"
    border_soft = "#132535" if dark else "#e8f2f7"
    selected = "#173d61" if dark else PALETTE["blue_soft"]
    selection_text = "#ffffff" if dark else "#102536"
    input_bg = "#09111a" if dark else "#ffffff"
    grade_bg = "#c4dffc"
    support_bg = "#0b2426" if dark else "#effaf8"
    limit_bg = "#251724" if dark else PALETTE["pink_soft"]
    support_text = "#8edbd6" if dark else "#236e75"
    limit_text = "#ffc5e4" if dark else "#8a3f68"
    code_bg = "#020508" if dark else "#fbfdff"

    return f"""
    QWidget {{
        background: {bg}; color: {text};
        font-family: 'Microsoft YaHei UI'; font-size: 13px;
    }}
    QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
    QMainWindow, QDialog {{ background: {bg}; }}
    QLabel#brand {{ font-size: 19px; font-weight: 750; color: {text}; }}
    QLabel#sectionTitle {{ font-size: 15px; font-weight: 700; }}
    QLabel#viewerTitle {{ font-size: 14px; font-weight: 700; }}
    QLabel#subheading {{ font-size: 13px; font-weight: 700; }}
    QLabel#muted {{ color: {muted}; }}
    QLabel#countBadge {{
        background: {PALETTE['pink_soft'] if not dark else '#3a2032'};
        color: {PALETTE['blue_deep'] if not dark else PALETTE['pink']};
        border-radius: 9px; min-width: 22px; padding: 2px 7px; font-weight: 700;
    }}

    QFrame#topbar {{ background: {('#11171e' if dark else '#ffffff')}; border-bottom: 1px solid {border}; }}
    QLabel#topMeta {{ color: {('#aeb9c4' if dark else '#546573')}; font-size: 11px; }}
    QLabel#activityDot {{ color: {PALETTE['accent']}; font-size: 13px; }}
    QFrame#topDivider {{ color: {border}; border: none; border-left: 1px solid {border}; max-width: 1px; }}
    QFrame#sidebar {{ background: {surface}; border-right: 1px solid {border}; }}
    QFrame#viewerPanel {{ background: {bg}; }}
    QFrame#viewerHeader {{ background: {surface}; border-bottom: 1px solid {border}; }}
    QFrame#inspector {{ background: {surface}; border-left: 1px solid {border}; }}
    QFrame#inspectorActions {{ background: {surface}; border-top: 1px solid {border}; }}
    QFrame#activityPanel {{ background: {surface}; border-top: 1px solid {border}; }}
    QLabel#activityTitle {{ font-size: 12px; font-weight: 700; color: {PALETTE['blue'] if dark else PALETTE['blue_deep']}; }}

    QPushButton, QComboBox {{
        background: {surface_alt}; color: {text}; border: 1px solid {border};
        border-radius: 7px; padding: 7px 12px;
    }}
    QPushButton:hover, QComboBox:hover {{ border-color: {PALETTE['blue']}; }}
    QPushButton:disabled {{ color: {muted}; background: {surface}; border-color: {border_soft}; }}
    QPushButton#primary {{ background: {PALETTE['blue_deep']}; color: white; border-color: {PALETTE['blue_deep']}; font-weight: 700; }}
    QPushButton#primary:hover {{ background: {PALETTE['blue']}; color: #102536; }}
    QPushButton#topAction {{
        background: {('#1a222b' if dark else '#f7fafc')}; color: {text};
        border: 1px solid {('#35404c' if dark else '#cddce7')}; border-radius: 5px;
        padding: 6px 12px; font-weight: 400;
    }}
    QPushButton#topAction:hover {{ border-color: {PALETTE['accent']}; background: {('#202b36' if dark else '#edf7fd')}; }}
    QPushButton#topPrimary {{
        background: {PALETTE['accent']}; color: #102536;
        border: 1px solid {PALETTE['accent']}; border-radius: 5px;
        padding: 6px 12px; font-weight: 400;
    }}
    QPushButton#topPrimary:hover {{ background: #d8effd; border-color: #d8effd; }}
    QPushButton#themeToggle {{
        background: transparent; color: {PALETTE['accent'] if dark else PALETTE['blue_deep']};
        border: 1px solid {border}; border-radius: 5px; padding: 4px 9px;
    }}
    QPushButton#softBlue {{
        background: {('#142b42' if dark else PALETTE['blue_soft'])};
        color: {('#dcedff' if dark else '#173a55')}; border-color: {PALETTE['blue']};
    }}
    QPushButton#quiet, QPushButton#toolButton {{ background: transparent; border-color: {border}; }}
    QPushButton#toolButton {{ padding: 5px 9px; }}
    QPushButton#correct {{
        background: {('#3a2032' if dark else PALETTE['pink_mid'])};
        color: {('#ffeaf5' if dark else '#66304f')}; border-color: {PALETTE['pink']}; font-weight: 700;
    }}
    QComboBox#themeSwitch {{ min-width: 88px; }}

    QLineEdit {{
        background: {input_bg}; border: 1px solid {border}; border-radius: 7px;
        padding: 7px 9px; selection-background-color: {PALETTE['blue']};
    }}
    QLineEdit:focus {{ border: 1px solid {PALETTE['blue']}; }}
    QCheckBox {{ color: {muted}; spacing: 7px; }}

    QTreeWidget, QListWidget, QTableWidget {{ background: {surface}; border: none; outline: none; }}
    QTreeWidget {{ font-size: 12px; }}
    QTreeWidget::item {{ padding: 2px 2px; border-radius: 3px; margin: 0; }}
    QTreeWidget::item:hover {{ background: {surface_alt}; }}
    QTreeWidget::item:selected {{ background: {selected}; color: {selection_text}; }}
    QListWidget::item {{ padding: 8px; border-bottom: 1px solid {border_soft}; }}
    QHeaderView::section {{
        background: {surface_alt}; color: {muted}; padding: 5px 8px;
        border: none; border-bottom: 1px solid {border};
    }}

    QTabWidget::pane {{ background: {surface}; border: none; }}
    QTabBar::tab {{ background: transparent; color: {muted}; padding: 7px 12px; border-bottom: 2px solid transparent; }}
    QTabBar::tab:selected {{ color: {PALETTE['blue'] if dark else PALETTE['blue_deep']}; border-bottom-color: {PALETTE['pink']}; font-weight: 700; }}

    QScrollArea {{ background: {bg}; border: none; }}
    QLabel#imageCanvas, QLabel#emptyViewer {{ background: {bg}; color: {muted}; font-size: 15px; }}
    QPlainTextEdit#codeViewer {{
        background: {code_bg}; color: {text}; border: none; padding: 14px;
        font-family: 'Cascadia Mono', 'Consolas'; font-size: 12px;
        selection-background-color: {selected};
    }}
    QTextBrowser#reportViewer {{
        background: {bg}; color: {text}; border: none; padding: 18px;
        font-family: 'Microsoft YaHei UI'; font-size: 13px;
        selection-background-color: {selected};
    }}

    QFrame#gradeCard {{ background: {grade_bg}; border: none; border-radius: 12px; }}
    QLabel#gradeKicker {{ color: #53616f; font-size: 11px; font-weight: 700; }}
    QLabel#grade {{ font-size: 46px; font-weight: 800; color: #122231; }}
    QLabel#gradePill {{
        background: #ffeef8;
        color: #66304f; border: 1px solid #fdc0e1;
        border-radius: 20px; min-width: 58px; min-height: 42px; padding: 4px 7px; font-weight: 700;
    }}
    QLabel#reviewWarning {{
        background: {limit_bg}; color: {limit_text}; border-left: 3px solid {PALETTE['pink']};
        border-radius: 5px; padding: 8px;
    }}
    QFrame#sectionBlock {{ background: transparent; border: none; }}
    QFrame#infoBox, QLabel#auditBox {{ background: {('#0d141e' if dark else '#f3f8fc')}; border: none; border-radius: 10px; padding: 0; }}
    QLabel#auditBox {{ padding: 10px; }}
    QFrame#inspector QPushButton#primary {{ background: #c4dffc; color: #122231; border-color: #c4dffc; }}
    QFrame#inspector QPushButton#correct {{ background: #ffeef8; color: #66304f; border-color: #fdc0e1; }}

    QTableWidget {{ background: {surface}; alternate-background-color: {surface_alt}; }}
    QSplitter::handle {{ background: {border}; }}
    QSplitter::handle:horizontal {{ width: 4px; }}
    QSplitter::handle:vertical {{ height: 5px; }}
    QSplitter::handle:hover {{ background: {PALETTE['blue']}; }}
    QScrollBar:vertical {{ background: transparent; width: 9px; margin: 0; }}
    QScrollBar::handle:vertical {{ background: {('rgba(190, 198, 207, 105)' if dark else 'rgba(96, 106, 116, 90)')}; min-height: 28px; border-radius: 4px; }}
    QScrollBar::handle:vertical:hover {{ background: {('rgba(205, 212, 220, 155)' if dark else 'rgba(82, 92, 102, 135)')}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; background: transparent; border: none; }}
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 0; }}
    QScrollBar::handle:horizontal {{ background: {('rgba(190, 198, 207, 105)' if dark else 'rgba(96, 106, 116, 90)')}; min-width: 28px; border-radius: 4px; }}
    QScrollBar::handle:horizontal:hover {{ background: {('rgba(205, 212, 220, 155)' if dark else 'rgba(82, 92, 102, 135)')}; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; background: transparent; border: none; }}
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: transparent; }}
    QStatusBar {{ background: {surface}; color: {muted}; border-top: 1px solid {border_soft}; }}
    QToolTip {{ background: {surface_high}; color: {text}; border: 1px solid {PALETTE['blue']}; padding: 5px; }}
    """
