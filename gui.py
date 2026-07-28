"""
视频转 3D 高斯溅射 — PySide6 图形界面

设计语言：深邃蓝灰背景 + 明亮青蓝强调色 + 清晰层次
"""

import os
import sys
import time
import threading
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QProgressBar, QTextEdit,
    QScrollArea, QMessageBox, QStackedWidget, QFormLayout, QListWidget,
    QFrame, QGraphicsDropShadowEffect, QCheckBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QPoint, QPropertyAnimation
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPainter

# ---------- Import simplified modules ----------
from frames import extract_frames
from poses import estimate_poses, CameraPose
from point_cloud import initialize_gaussians
from gaussian import Gaussian3D, DifferentiableRasterizer, Trainer, LazyFrames, LossDivergenceError
from exporter import export_training_checkpoint


# ============================================================================
# 设计系统
# ============================================================================

C = {
    "bg":             "#0f1724",
    "bg_sidebar":     "#141e2c",
    "bg_panel":       "#192436",
    "bg_card":        "#1c2a3e",
    "bg_input":       "#0f1724",
    "border":         "#253346",
    "border_light":   "#1e2a38",
    "border_focus":   "#3498db",
    "text_primary":   "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted":     "#6e7681",
    "accent":         "#3498db",
    "accent_hover":   "#5dade2",
    "success":        "#2ecc71",
    "success_bg":     "#239b56",
    "danger":         "#e74c3c",
    "danger_bg":      "#c0392b",
    "purple":         "#9b59b6",
    "title":          "#ffffff",
}

RADIUS_CARD = 12
RADIUS_INPUT = 8
RADIUS_BTN = 8


def card_bg():
    return (
        f"background-color: {C['bg_card']}; "
        f"border: 1px solid {C['border']}; "
        f"border-radius: {RADIUS_CARD}px;"
    )


def input_fg():
    return (
        f"background-color: {C['bg_input']}; "
        f"color: {C['text_primary']}; "
        f"border: 1px solid {C['border']}; "
        f"border-radius: {RADIUS_INPUT}px; "
        f"padding: 6px 12px; font-size: 12px;"
    )


# ============================================================================
# 自定义控件 (保持原有风格，未大幅改动)
# ============================================================================

class RoundedCard(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(card_bg())
        self.setContentsMargins(0, 0, 0, 0)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)


class StyledLabel(QLabel):
    def __init__(self, text="", font_size=11, color=None, bold=False, parent=None):
        super().__init__(text, parent)
        c = color or C["text_secondary"]
        w = "bold;" if bold else ""
        self.setStyleSheet(
            f"color: {c}; font-size: {font_size}pt; font-weight: {w}; "
            f"font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;"
        )


class StyledSpinBox(QWidget):
    valueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self._min_val = 0
        self._max_val = 9999
        self._single_step = 1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        btn_style = f"""
            QPushButton {{
                background-color: transparent; color: {C['text_secondary']}; border: none;
                border-radius: 6px; font-size: 16px; font-weight: bold; padding: 0; line-height: 1;
            }}
            QPushButton:hover {{
                background-color: {C['border_light']}; color: {C['accent']};
            }}
        """
        self._btn_down = QPushButton("−", self)
        self._btn_down.setFixedSize(24, 28)
        self._btn_down.setCursor(Qt.PointingHandCursor)
        self._btn_down.setStyleSheet(btn_style)
        self._btn_down.clicked.connect(self._decrement)

        self._edit = QLineEdit(self)
        self._edit.setAlignment(Qt.AlignCenter)
        self._edit.setFixedWidth(56)
        self._edit.setMaxLength(10)
        self._edit.setText(str(self._value))
        self._edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: {C['bg_input']}; color: {C['text_primary']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 2px 4px; font-size: 12px;
                font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
            }}
            QLineEdit:focus {{ border: 1px solid {C['border_focus']}; }}
        """)
        self._edit.returnPressed.connect(self._commit_value)

        self._btn_up = QPushButton("+", self)
        self._btn_up.setFixedSize(24, 28)
        self._btn_up.setCursor(Qt.PointingHandCursor)
        self._btn_up.setStyleSheet(btn_style)
        self._btn_up.clicked.connect(self._increment)

        layout.addWidget(self._btn_down)
        layout.addWidget(self._edit)
        layout.addWidget(self._btn_up)

    def _increment(self):
        self._set_value(min(self._value + self._single_step, self._max_val))

    def _decrement(self):
        self._set_value(max(self._value - self._single_step, self._min_val))

    def _set_value(self, val):
        if val != self._value:
            self._value = val
            self._edit.setText(str(val))
            self.valueChanged.emit(val)

    def _commit_value(self):
        try:
            val = int(self._edit.text())
            self._set_value(max(self._min_val, min(self._max_val, val)))
        except ValueError:
            self._edit.setText(str(self._value))

    def setValue(self, val):
        self._set_value(max(self._min_val, min(self._max_val, val)))

    def value(self):
        return self._value

    def setRange(self, lo, hi):
        self._min_val = lo
        self._max_val = hi

    def setSingleStep(self, step):
        self._single_step = step


class StyledDoubleSpinBox(QWidget):
    valueChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._min_val = 0.0
        self._max_val = 9999.0
        self._single_step = 1.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        btn_style = f"""
            QPushButton {{
                background-color: transparent; color: {C['text_secondary']}; border: none;
                border-radius: 6px; font-size: 16px; font-weight: bold; padding: 0; line-height: 1;
            }}
            QPushButton:hover {{
                background-color: {C['border_light']}; color: {C['accent']};
            }}
        """
        self._btn_down = QPushButton("−")
        self._btn_down.setFixedSize(24, 28)
        self._btn_down.setCursor(Qt.PointingHandCursor)
        self._btn_down.setStyleSheet(btn_style)
        self._btn_down.clicked.connect(self._decrement)

        self._edit = QLineEdit()
        self._edit.setAlignment(Qt.AlignCenter)
        self._edit.setFixedWidth(64)
        self._edit.setMaxLength(12)
        self._edit.setText(str(self._value))
        self._edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: {C['bg_input']}; color: {C['text_primary']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 2px 4px; font-size: 12px;
                font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
            }}
            QLineEdit:focus {{ border: 1px solid {C['border_focus']}; }}
        """)
        self._edit.returnPressed.connect(self._commit_value)

        self._btn_up = QPushButton("+")
        self._btn_up.setFixedSize(24, 28)
        self._btn_up.setCursor(Qt.PointingHandCursor)
        self._btn_up.setStyleSheet(btn_style)
        self._btn_up.clicked.connect(self._increment)

        layout.addWidget(self._btn_down)
        layout.addWidget(self._edit)
        layout.addWidget(self._btn_up)

    def _increment(self):
        self._set_value(min(self._value + self._single_step, self._max_val))

    def _decrement(self):
        self._set_value(max(self._value - self._single_step, self._min_val))

    def _set_value(self, val):
        if abs(val - self._value) > 1e-10:
            self._value = val
            self._edit.setText(f"{val:g}")
            self.valueChanged.emit(val)

    def _commit_value(self):
        try:
            val = float(self._edit.text())
            self._set_value(max(self._min_val, min(self._max_val, val)))
        except ValueError:
            self._edit.setText(str(self._value))

    def setValue(self, val):
        self._set_value(max(self._min_val, min(self._max_val, val)))

    def value(self):
        return self._value

    def setRange(self, lo, hi):
        self._min_val = lo
        self._max_val = hi

    def setSingleStep(self, step):
        self._single_step = step


class StyledComboBox(QWidget):
    currentTextChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._current_index = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._display = QLabel("—")
        self._display.setStyleSheet(f"""
            QLabel {{
                background-color: {C['bg_input']}; color: {C['text_primary']};
                border: 1px solid {C['border']};
                border-top-left-radius: 6px; border-bottom-left-radius: 6px;
                padding: 4px 10px; font-size: 12px;
                font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
            }}
        """)
        self._display.setAlignment(Qt.AlignVCenter)
        layout.addWidget(self._display, stretch=1)

        self._arrow_btn = QPushButton("▼")
        self._arrow_btn.setFixedSize(28, 30)
        self._arrow_btn.setCursor(Qt.PointingHandCursor)
        self._arrow_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent; color: {C['text_secondary']};
                border: 1px solid {C['border']};
                border-top-right-radius: 6px; border-bottom-right-radius: 6px;
                font-size: 12px; padding: 0;
            }}
            QPushButton:hover {{
                background-color: {C['border_light']}; color: {C['accent']};
            }}
        """)
        self._arrow_btn.clicked.connect(self._toggle_popup)
        layout.addWidget(self._arrow_btn)

        self._popup = QFrame(None, Qt.Popup | Qt.FramelessWindowHint)
        self._popup.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self._popup.setStyleSheet(f"QFrame {{ background-color: {C['bg_card']}; border: 1px solid {C['border']}; border-radius: {RADIUS_CARD}px; }}")
        pl = QVBoxLayout(self._popup)
        pl.setContentsMargins(2, 2, 2, 2)
        pl.setSpacing(0)

        self._list_widget = QListWidget()
        self._list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {C['bg_card']}; color: {C['text_primary']};
                border: none; font-size: 12px;
                font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
                padding: 2px;
            }}
            QListWidget::item {{ min-height: 28px; padding: 2px 10px; border-radius: 4px; margin: 1px 4px; }}
            QListWidget::item:selected {{ background-color: {C['accent']}; color: #fff; }}
            QListWidget::item:hover:!selected {{ background-color: {C['bg_input']}; color: {C['text_primary']}; }}
        """)
        self._list_widget.currentRowChanged.connect(self._on_select)
        pl.addWidget(self._list_widget)

        self._anim = QPropertyAnimation(self._popup, b"windowOpacity")
        self._anim.setDuration(120)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._is_open = False

    def addItems(self, items):
        self._items = list(items)
        self._list_widget.clear()
        for item in items:
            self._list_widget.addItem(item)
        if items:
            self.setCurrentIndex(0)

    def setCurrentIndex(self, idx):
        if 0 <= idx < len(self._items):
            self._current_index = idx
            self._display.setText(self._items[idx])
            self._list_widget.setCurrentRow(idx)

    def currentIndex(self):
        return self._current_index

    def currentText(self):
        return self._items[self._current_index] if 0 <= self._current_index < len(self._items) else ""

    def _toggle_popup(self):
        if self._is_open:
            self._close_popup()
        else:
            self._open_popup()

    def _open_popup(self):
        scr = QApplication.primaryScreen().geometry()
        pos = self.mapToGlobal(QPoint(0, self.height()))
        self._popup.show()
        self._popup.raise_()
        self._popup.activateWindow()

        ih = self._list_widget.sizeHintForRow(0) + 4
        v = min(len(self._items), 5)
        ph = v * ih + 8
        self._popup.resize(self._display.width() + 28, ph)

        x, y = pos.x(), pos.y()
        if x + self._popup.width() > scr.right():
            x = scr.right() - self._popup.width()
        if y + self._popup.height() > scr.bottom():
            y = pos.y() - self._popup.height()
        if y < scr.top():
            y = pos.y()
        self._popup.move(x, y)

        self._popup.setWindowOpacity(0.0)
        self._anim.start()
        self._is_open = True
        self._list_widget.setFocus(Qt.PopupFocusReason)

    def _close_popup(self):
        self._popup.hide()
        self._is_open = False

    def _on_select(self, row):
        if 0 <= row < len(self._items):
            self._current_index = row
            self._display.setText(self._items[row])
            self.currentTextChanged.emit(self._items[row])
            self._close_popup()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self._is_open:
            self._close_popup()
        elif e.key() in (Qt.Key_Down, Qt.Key_Up, Qt.Key_Enter, Qt.Key_Space) and not self._is_open:
            self._open_popup()
        else:
            super().keyPressEvent(e)


class StyledButton(QPushButton):
    def __init__(self, text, bg_color=C["bg_card"], hover_color=None, radius=RADIUS_BTN, font_size=13, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        hc = hover_color or bg_color
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color}; color: {C["text_primary"]}; border: none;
                border-radius: {radius}px; font-size: {font_size}px; font-weight: bold;
                padding: 8px 20px; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                border-bottom: 2px solid rgba(0,0,0,0.3);
            }}
            QPushButton:hover {{ background-color: {hc}; }}
            QPushButton:pressed {{ background-color: {bg_color}; opacity: 0.9; border-bottom: 1px solid rgba(0,0,0,0.3); }}
            QPushButton:disabled {{ background-color: {C["border_light"]}; color: {C["text_muted"]}; border-bottom: 2px solid transparent; }}
        """)


class AccentButton(StyledButton):
    def __init__(self, text, parent=None):
        super().__init__(text, C["accent"], C["accent_hover"], parent=parent)


class SuccessButton(StyledButton):
    def __init__(self, text, parent=None):
        super().__init__(text, "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #239b56, stop:1 #2ecc71)", "#27ae60", parent=parent)


class DangerButton(StyledButton):
    def __init__(self, text, parent=None):
        super().__init__(text, "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #c0392b, stop:1 #e74c3c)", "#e74c3c", parent=parent)


class ThinProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)
        self.setTextVisible(False)
        self.setValue(0)
        self.setStyleSheet(f"""
            QProgressBar {{ background-color: {C["border_light"]}; border: none; border-radius: 3px; }}
            QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {C["accent"]}, stop:1 {C["purple"]}); border-radius: 3px; }}
        """)


class StatusDot(QWidget):
    def __init__(self, status="idle", parent=None):
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self.status = status
        self._colors = {"idle": C["text_muted"], "running": C["success"], "error": C["danger"], "done": C["success"]}
        self.color = self._colors[status]

    def set_status(self, s):
        self.status = s
        self.color = self._colors.get(s, C["text_muted"])
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.color))
        p.drawEllipse(1, 1, 8, 8)


class LogViewer(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {C["bg"]}; color: {C["text_secondary"]};
                border: 1px solid {C["border_light"]}; border-radius: {RADIUS_INPUT}px;
                padding: 10px 14px; line-height: 1.6;
            }}
        """)

    def append_log(self, msg):
        self.append(msg)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


class PreviewImage(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setMaximumSize(560, 380)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"QLabel {{ background-color: {C['bg']}; border: 2px dashed {C['border_light']}; border-radius: {RADIUS_CARD}px; color: {C['text_muted']}; font-size: 13px; }}")
        self.setText("📷  选择视频以预览")

    def set_image(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = QImage(img.tobytes(), w, h, w * c, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(self.width() - 4, self.height() - 4, Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ============================================================================
# 损失曲线控件
# ============================================================================

class LossCurvePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        layout.addWidget(StyledLabel("📈 损失曲线", font_size=14, bold=True, color=C["title"]))

        fc = RoundedCard()
        fl = QVBoxLayout(fc)
        fl.setContentsMargins(16, 14, 16, 14)
        fl.setSpacing(10)
        fl.addWidget(StyledLabel("本轮每帧损失", font_size=12, bold=True, color=C["text_secondary"]))
        self.frame_image = QLabel()
        self.frame_image.setAlignment(Qt.AlignCenter)
        self.frame_image.setMinimumHeight(240)
        self.frame_image.setStyleSheet(f"background-color: {C['bg']}; border: 1px solid {C['border_light']}; border-radius: {RADIUS_CARD}px; color: {C['text_muted']};")
        self.frame_image.setText("暂无数据")
        fl.addWidget(self.frame_image, stretch=1)
        layout.addWidget(fc, stretch=1)

        ec = RoundedCard()
        el = QVBoxLayout(ec)
        el.setContentsMargins(16, 14, 16, 14)
        el.setSpacing(10)
        el.addWidget(StyledLabel("每轮平均损失", font_size=12, bold=True, color=C["text_secondary"]))
        self.epoch_image = QLabel()
        self.epoch_image.setAlignment(Qt.AlignCenter)
        self.epoch_image.setMinimumHeight(240)
        self.epoch_image.setStyleSheet(f"background-color: {C['bg']}; border: 1px solid {C['border_light']}; border-radius: {RADIUS_CARD}px; color: {C['text_muted']};")
        self.epoch_image.setText("暂无数据")
        el.addWidget(self.epoch_image, stretch=1)
        layout.addWidget(ec, stretch=1)

        self._mpl_style = {
            "figure.facecolor": C["bg"],
            "axes.facecolor": C["bg"],
            "axes.edgecolor": C["border"],
            "axes.labelcolor": C["text_secondary"],
            "xtick.color": C["text_muted"],
            "ytick.color": C["text_muted"],
            "grid.color": C["border"],
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei UI", "Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 4,
        }

    def update_from_data(self, data):
        if data.get("type") == "frame":
            self._render_frame_chart(data.get("data", []))
        elif data.get("type") == "epoch":
            self._render_epoch_chart(data.get("data", []))

    def _render_frame_chart(self, losses):
        with plt.style.context(self._mpl_style):
            fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
            fig.patch.set_facecolor(C["bg"])
            ax.set_facecolor(C["bg"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if not losses:
                ax.text(0.5, 0.5, "暂无数据", transform=ax.transAxes, ha="center", va="center", color=C["text_muted"], fontsize=12)
            else:
                xs = [l[1] for l in losses]
                ys = [l[2] for l in losses]
                if len(xs) > 500:
                    step = max(1, len(xs) // 500)
                    xs_s = xs[::step]
                    ys_s = ys[::step]
                else:
                    xs_s, ys_s = xs, ys
                ax.plot(xs_s, ys_s, color="#3498db", linewidth=2.0, alpha=0.9, marker=".", markersize=2, markeredgewidth=0)
                ax.fill_between(xs_s, ys_s, alpha=0.15, color="#3498db")
                ax.set_xlabel("帧索引", color=C["text_secondary"], fontsize=11, labelpad=8)
                ax.set_ylabel("损失值", color=C["text_secondary"], fontsize=11, labelpad=8)
                ax.set_title(f"当前轮次帧损失 · {len(losses)} 帧", color=C["title"], fontsize=13, pad=12)
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
                if ys:
                    ax.set_ylim(max(0, np.min(ys) * 0.9), np.percentile(ys, 98) * 1.1)

            fig.tight_layout()
            self._set_pixmap(self.frame_image, fig)
            plt.close(fig)

    def _render_epoch_chart(self, losses):
        with plt.style.context(self._mpl_style):
            fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
            fig.patch.set_facecolor(C["bg"])
            ax.set_facecolor(C["bg"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if not losses:
                ax.text(0.5, 0.5, "暂无数据", transform=ax.transAxes, ha="center", va="center", color=C["text_muted"], fontsize=12)
            else:
                epochs = [l[0] for l in losses]
                values = [l[1] for l in losses]
                ax.plot(epochs, values, color="#9b59b6", linewidth=2.4, alpha=0.9, marker="o", markersize=6, markeredgewidth=0, markerfacecolor="#b388ff")
                ax.fill_between(epochs, values, alpha=0.12, color="#9b59b6")
                ax.set_xlabel("训练轮次", color=C["text_secondary"], fontsize=11, labelpad=8)
                ax.set_ylabel("平均损失", color=C["text_secondary"], fontsize=11, labelpad=8)
                ax.set_title(f"轮次平均损失 · {len(epochs)} 轮", color=C["title"], fontsize=13, pad=12)
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
                if values:
                    for i in range(max(0, len(values) - 5), len(values)):
                        ax.annotate(f"{values[i]:.4f}", (epochs[i], values[i]), textcoords="offset points", xytext=(0, 14),
                                    ha='center', fontsize=8, color=C["text_muted"],
                                    arrowprops=dict(arrowstyle="-", color=C["border_light"], lw=0.6))

            fig.tight_layout()
            self._set_pixmap(self.epoch_image, fig)
            plt.close(fig)

    def _set_pixmap(self, label, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        img = QImage()
        if img.loadFromData(buf.read(), "PNG"):
            label.setPixmap(QPixmap.fromImage(img).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            label.setText("图片加载失败")
        buf.close()


# ============================================================================
# 工作线程
# ============================================================================

class PipelineWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)
    frame_paths_signal = Signal(list)
    loss_signal = Signal(dict)

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._running = True
        self._stop_event = threading.Event()
        self._current_epoch = 0
        self._current_epoch_frame_losses = []
        self._all_epoch_losses = []

    def stop(self):
        self._running = False
        self._stop_event.set()

    def _log(self, msg):
        if self._running:
            self.log_signal.emit(msg)

    def _set_progress(self, pct, msg):
        if self._running:
            self.progress_signal.emit(pct, msg)

    def _emit_frame_loss(self, epoch, index, loss):
        if not self._running:
            return
        self._current_epoch = epoch
        self._current_epoch_frame_losses.append((epoch, index, float(loss)))
        self.loss_signal.emit({"type": "frame", "data": list(self._current_epoch_frame_losses)})

    def _emit_epoch_loss(self, epoch, avg_loss):
        if not self._running:
            return
        self._all_epoch_losses.append((epoch, float(avg_loss)))
        self.loss_signal.emit({"type": "epoch", "data": list(self._all_epoch_losses)})

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            import traceback
            self._log(f"错误: {e}")
            self._log(traceback.format_exc())
            self.finished_signal.emit(False, str(e))

    def _run_pipeline(self):
        c = self.config
        workdir = Path(c["workdir"])
        frame_dir = workdir / "frames"
        poses_dir = workdir / "poses"

        # Check for resume
        has_frames = (workdir / "frame_paths.txt").exists() and frame_dir.exists() and any(frame_dir.iterdir())
        has_poses = (workdir / "intrinsics.npy").exists() and (workdir / "sparse_points.npy").exists()
        has_gaussians = (workdir / "gaussian_params.npz").exists()
        has_training_state = (workdir / "training_state.pt").exists()
        can_resume = has_frames and has_poses and has_gaussians

        # ---------- Step 1: Extract Frames ----------
        self._log("[1/5] 正在提取视频帧...")
        self._set_progress(0, "正在提取视频帧...")

        if has_frames:
            frame_paths = [p.strip() for p in (workdir / "frame_paths.txt").read_text().splitlines()]
            self._log(f"  已加载 {len(frame_paths)} 帧（跳过提取）")
        else:
            smart_sampling = c["sampling_mode"] != "uniform"
            two_stage = c["sampling_mode"] == "two-stage"

            frame_paths = extract_frames(
                video_path=c["video"],
                output_dir=str(frame_dir),
                fps=c["fps"],
                scale=c["scale"],
                min_frames=c["min_frames"],
                max_frames=c["max_frames"],
                smart_sampling=smart_sampling,
                two_stage=two_stage,
                poses_output_dir=str(poses_dir / "coarse_poses") if two_stage else None,
                optical_flow_method="farneback",
            )
            (workdir / "frame_paths.txt").write_text("\n".join(frame_paths))
            self._log(f"  已提取 {len(frame_paths)} 帧")
            self.frame_paths_signal.emit(frame_paths)

        self._set_progress(10, f"{len(frame_paths)} 帧就绪")
        frames = LazyFrames(frame_paths)
        h, w = frames[0].shape[:2]
        self._log(f"  分辨率: {w}x{h}")

        # ---------- Step 2: Estimate Poses ----------
        self._log("\n[2/5] 正在估算相机姿态...")
        self._set_progress(20, "正在估算相机姿态...")

        if has_poses:
            K = np.load(workdir / "intrinsics.npy")
            poses_data = np.load(workdir / "poses.npy")
            sparse_points = np.load(workdir / "sparse_points.npy")
            poses = [CameraPose(R=p[:3, :3].copy(), t=p[:3, 3].copy()) for p in poses_data]
            self._log(f"  已加载 {len(poses)} 个姿态（跳过估算）")
        else:
            if c["pose_estimator"] == "colmap":
                try:
                    from colmap_poses import estimate_poses_with_colmap
                    self._log("  使用 COLMAP 进行姿态估算...")
                    intrinsics, poses, sparse_points = estimate_poses_with_colmap(
                        frame_paths, str(workdir)
                    )
                    K = intrinsics.K
                except (ImportError, RuntimeError) as e:
                    self._log(f"  [ERROR] COLMAP 失败: {e}")
                    self.finished_signal.emit(False, f"COLMAP failed: {e}")
                    return
            else:
                self._log("  使用 ORB+EM 进行姿态估算...")
                intrinsics, poses, sparse_points = estimate_poses(
                    frame_paths,
                    str(poses_dir),
                    min_inliers=25,
                    feature_type="orb",
                    focal_guess=c.get("focal_guess"),
                    aspect_ratio=1.0,
                )
                K = intrinsics.K

            np.save(workdir / "intrinsics.npy", K)
            valid_poses = [p for p in poses if p is not None]
            if valid_poses:
                np.save(workdir / "poses.npy", np.stack([p.RT for p in valid_poses]))
            if sparse_points is not None and sparse_points.size > 0:
                np.save(workdir / "sparse_points.npy", sparse_points)

        # Ensure poses list length matches frames
        while len(poses) < len(frame_paths):
            poses.append(None)
        valid_count = sum(1 for p in poses if p is not None)
        self._log(f"  {valid_count} 个有效姿态 (共 {len(frame_paths)} 帧)")
        self._set_progress(30, "姿态估算完成")

        if valid_count < 3:
            self._log("  错误: 有效姿态太少，请检查视频质量")
            self.finished_signal.emit(False, "Too few valid poses")
            return

        # ---------- Step 3: Initialize Gaussians ----------
        self._log("\n[3/5] 正在初始化3D高斯...")
        self._set_progress(35, "正在初始化高斯...")

        if has_gaussians:
            params = dict(np.load(workdir / "gaussian_params.npz"))
            gauss_init = {k: params[k] for k in ["positions", "scales", "opacities", "sh_coeffs", "rotations"]}
            self._log(f"  已加载 {params['positions'].shape[0]} 个高斯（跳过初始化）")
        else:
            class _I:
                pass
            _i = _I()
            _i.K = K
            gauss_init = initialize_gaussians(
                sparse_points=sparse_points,
                poses=poses,
                frame_paths=frame_paths,
                intrinsics=_i,
            )
            np.savez(workdir / "gaussian_params.npz", **gauss_init)

        num_gs = gauss_init["positions"].shape[0]
        self._log(f"  共 {num_gs} 个高斯")
        self._set_progress(40, f"{num_gs} 个高斯就绪")

        # ---------- Step 4: Training ----------
        self._log(f"\n[4/5] 正在训练 ({c['device']}，{c['num_epochs']} 轮)...")
        self._set_progress(45, "正在训练...")

        gaussians = Gaussian3D()
        gaussians.initialize_from_dict(gauss_init, device=c["device"])
        rasterizer = DifferentiableRasterizer(image_width=w, image_height=h)

        trainer = Trainer(
            gaussians=gaussians,
            rasterizer=rasterizer,
            K=K,
            image_width=w,
            image_height=h,
            device=c["device"],
            use_cuda_rasterizer=True,
            sh_degree=c["sh_degree"],
            random_background=c["random_background"],
            train_focal=c["train_focal"],
            max_gaussians=c["max_gaussians"],
            sh_warmup_steps=c["sh_warmup_steps"],
            ssim_warmup_steps=c["ssim_warmup_steps"],
            ssim_weight_max=c["ssim_weight_max"],
            enable_k1=c["enable_k1"],
        )

        train_poses = [p.RT.astype(np.float32) if p is not None else None for p in poses]
        start_epoch = 1
        pt_ckpt = str(workdir / "training_state.pt")
        best_loss = float("inf")
        training_start = time.time()

        # Resume if possible
        if can_resume and has_training_state:
            try:
                trainer.load_training_state(pt_ckpt, device=c["device"])
                saved = trainer.current_step
                resumed_epoch = max(1, saved // max(len(frame_paths), 1))
                self._log(f"  已恢复训练状态: {saved} 帧已训练，从第 {resumed_epoch} 轮继续")
                start_epoch = resumed_epoch
            except Exception as e:
                self._log(f"  [WARN] 加载检查点失败: {e}，从头开始")

        for epoch in range(start_epoch, c["num_epochs"] + 1):
            if not self._running:
                self._log("  用户已取消训练")
                trainer.save_training_state(pt_ckpt)
                self.finished_signal.emit(False, "已取消")
                return

            try:
                def _prog(fi, tot, loss):
                    self._log(f"[{epoch}/{c['num_epochs']}] 帧 {fi}/{tot} | 损失: {loss:.6f}")
                    pct = 50 + int((epoch - start_epoch) / max(1, c["num_epochs"] - start_epoch) * 40)
                    pct += min(10, int(fi / max(1, tot) * 10))
                    self.progress_signal.emit(min(pct, 94), f"帧 {fi}/{tot} | 损失: {loss:.6f}")
                    self._emit_frame_loss(epoch, fi, loss)

                avg_loss = trainer.train_epoch(
                    frames_iter=frame_paths,
                    camera_poses=train_poses,
                    stop_event=self._stop_event,
                    progress_callback=_prog,
                    loss_threshold=1.0,
                    checkpoint_path=pt_ckpt,
                )
            except LossDivergenceError as e:
                self._log(f"\n  [LOSS DIVERGENCE] {e}")
                trainer.save_training_state(pt_ckpt)
                self.finished_signal.emit(False, "损失发散")
                return
            except torch.cuda.OutOfMemoryError:
                self._log(f"\n  [OOM] CUDA 显存不足")
                self._log("  尝试降低高斯上限并修剪...")
                # Reduce max_gaussians and prune
                new_max = max(100000, trainer.adaptive_density.max_gaussians // 2)
                trainer.adaptive_density.max_gaussians = new_max
                n_pruned = trainer.adaptive_density.prune(min_opacity=0.05)
                self._log(f"  已修剪 {n_pruned} 个高斯，新上限 {new_max}")
                trainer.save_training_state(pt_ckpt)
                # Continue training with reduced budget
                avg_loss = trainer.train_epoch(
                    frames_iter=frame_paths,
                    camera_poses=train_poses,
                    stop_event=self._stop_event,
                    progress_callback=_prog,
                    loss_threshold=1.0,
                    checkpoint_path=pt_ckpt,
                )

            self._emit_epoch_loss(epoch, avg_loss)
            self._current_epoch_frame_losses = []

            elapsed = time.time() - training_start
            current_gs = trainer.gaussians.num_gaussians
            self._log(f"  轮次 {epoch:>5d}/{c['num_epochs']} | 损失: {avg_loss:.6f} | 耗时: {elapsed:.0f}s | 高斯数: {current_gs}")

            if avg_loss < best_loss:
                best_loss = avg_loss

            pct = 45 + int((epoch - start_epoch + 1) / max(1, c["num_epochs"] - start_epoch + 1) * 50)
            self._set_progress(min(pct, 95), f"训练轮次 {epoch}/{c['num_epochs']}")

        self._log(f"\n  训练完成。最佳损失: {best_loss:.6f}")
        self._set_progress(95, "训练完成")

        # ---------- Step 5: Export ----------
        self._log("\n[5/5] 正在导出 PLY...")
        self._set_progress(98, "正在导出...")

        export_training_checkpoint(trainer, c["output"], sh_degree=c["sh_degree"])

        self._log(f"\n完成！输出: {os.path.abspath(c['output'])}")
        self._set_progress(100, "完成！")
        self.finished_signal.emit(True, "成功")


# ============================================================================
# 主窗口
# ============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.setWindowTitle("3D 高斯溅射重建")
        self.resize(1300, 820)
        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ---- Sidebar ----
        sidebar = RoundedCard()
        sidebar.setFixedWidth(410)
        sidebar.setStyleSheet(f"background-color: {C['bg_sidebar']}; border-right: 1px solid {C['border']}; border-radius: 0;")
        sidebar.setGraphicsEffect(None)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Header
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 20, 20, 0)
        header_layout.setSpacing(4)

        self.title_label = StyledLabel("3D Gaussian Splatting", font_size=18, bold=True, color=C["title"])
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(StyledLabel("3D高斯溅射点云重建", font_size=11, color=C["text_secondary"]))

        sr = QHBoxLayout()
        self.status_dot = StatusDot("idle")
        self.status_text = StyledLabel("就绪", font_size=10, color=C["text_muted"])
        sr.addWidget(self.status_dot)
        sr.addWidget(self.status_text)
        sr.addStretch()
        header_layout.addLayout(sr)

        sep_top = QFrame()
        sep_top.setFixedHeight(1)
        sep_top.setStyleSheet(f"background-color: {C['border']}; margin: 8px 0;")
        header_layout.addWidget(sep_top)
        sidebar_layout.addWidget(header_widget)

        # Scrollable params
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.verticalScrollBar().setStyleSheet(f"""
            QScrollBar:vertical {{ background: {C['bg_sidebar']}; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 3px; min-height: 24px; }}
            QScrollBar::handle:vertical:hover {{ background: {C['text_secondary']}; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; }}
        """)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 12, 20, 20)
        scroll_layout.setSpacing(16)

        self._add_input_section(scroll_layout)
        self._add_param_section(scroll_layout)

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background-color: {C['border']}; margin: 4px 0;")
        scroll_layout.addWidget(sep2)

        btn_area = QWidget()
        bl = QVBoxLayout(btn_area)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)

        self.start_btn = SuccessButton("▶ 开始重建")
        self.start_btn.setFixedHeight(42)
        self.start_btn.clicked.connect(self._on_start)
        bl.addWidget(self.start_btn)

        self.stop_btn = DangerButton("■ 停止")
        self.stop_btn.setFixedHeight(42)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        bl.addWidget(self.stop_btn)

        scroll_layout.addWidget(btn_area)

        prog_area = QWidget()
        pl = QVBoxLayout(prog_area)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(4)
        self.progress_label = StyledLabel("就绪", font_size=10, color=C["text_secondary"])
        self.progress_bar = ThinProgressBar()
        pl.addWidget(self.progress_label)
        pl.addWidget(self.progress_bar)
        scroll_layout.addWidget(prog_area)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll)
        main_layout.addWidget(sidebar)

        # ---- Right panel ----
        ma = QWidget()
        rl = QVBoxLayout(ma)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        # Tabs
        tab_bar = QWidget()
        tbl = QHBoxLayout(tab_bar)
        tbl.setContentsMargins(20, 12, 20, 0)

        tab_style = """
            QPushButton#tabBtnActive {
                background-color: transparent; color: #e8ecf1;
                border-bottom: 2px solid #3498db;
                border-top: none; border-left: none; border-right: none;
                border-radius: 0; padding: 6px 16px; font-size: 13px; font-weight: bold;
            }
            QPushButton#tabBtnInactive {
                background-color: transparent; color: #8b949e;
                border-bottom: 2px solid transparent;
                border-top: none; border-left: none; border-right: none;
                border-radius: 0; padding: 6px 16px; font-size: 13px;
            }
            QPushButton#tabBtnInactive:hover { color: #e8ecf1; }
        """

        self.tab_btn_preview = StyledButton("🖼 帧预览", C["bg_panel"], C["border"], RADIUS_INPUT, 12)
        self.tab_btn_preview.setObjectName("tabBtnActive")
        self.tab_btn_preview.setStyleSheet(tab_style)
        self.tab_btn_preview.clicked.connect(lambda: self._switch_tab(0))

        self.tab_btn_logs = StyledButton("📋 运行日志", C["bg_panel"], C["border"], RADIUS_INPUT, 12)
        self.tab_btn_logs.setObjectName("tabBtnInactive")
        self.tab_btn_logs.setStyleSheet(tab_style)
        self.tab_btn_logs.clicked.connect(lambda: self._switch_tab(1))

        self.tab_btn_loss = StyledButton("📈 损失曲线", C["bg_panel"], C["border"], RADIUS_INPUT, 12)
        self.tab_btn_loss.setObjectName("tabBtnInactive")
        self.tab_btn_loss.setStyleSheet(tab_style)
        self.tab_btn_loss.clicked.connect(lambda: self._switch_tab(2))

        tbl.addWidget(self.tab_btn_preview)
        tbl.addWidget(self.tab_btn_logs)
        tbl.addWidget(self.tab_btn_loss)
        tbl.addStretch()
        rl.addWidget(tab_bar)

        ts = QFrame()
        ts.setFixedHeight(1)
        ts.setStyleSheet(f"background-color: {C['border']};")
        rl.addWidget(ts)

        self.stacked = QStackedWidget()
        self.stacked.setStyleSheet("background-color: transparent; border: none;")

        # Preview tab
        preview_page = QWidget()
        ppl = QVBoxLayout(preview_page)
        ppl.setContentsMargins(20, 16, 20, 16)
        ppl.setSpacing(12)
        self.preview_widget = PreviewImage()
        ppl.addWidget(self.preview_widget, alignment=Qt.AlignCenter)

        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        self.thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumb_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.thumb_inner = QWidget()
        self.thumb_layout = QVBoxLayout(self.thumb_inner)
        self.thumb_layout.setAlignment(Qt.AlignCenter)
        self.thumb_layout.setSpacing(8)
        self.thumb_scroll.setWidget(self.thumb_inner)
        ppl.addWidget(self.thumb_scroll)
        self.stacked.addWidget(preview_page)

        # Log tab
        log_page = QWidget()
        ll = QVBoxLayout(log_page)
        ll.setContentsMargins(20, 16, 20, 16)
        ll.setSpacing(0)
        self.log_viewer = LogViewer()
        ll.addWidget(self.log_viewer)
        self.stacked.addWidget(log_page)

        # Loss curve tab
        self.loss_curve_page = LossCurvePage()
        self.stacked.addWidget(self.loss_curve_page)

        rl.addWidget(self.stacked)
        rl.setStretchFactor(self.stacked, 1)
        main_layout.addWidget(ma, stretch=1)

    def _make_path_row(self, name, default=""):
        edit = QLineEdit(default)
        edit.setStyleSheet(f"{input_fg()} min-height: 32px;")
        btn = AccentButton("浏览")
        btn.setFixedWidth(64)
        btn.clicked.connect(lambda _, e=edit: self._browse_file(name, e))
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        row.setSpacing(8)
        setattr(self, f"{name}_edit", edit)
        return row

    def _add_input_section(self, parent_layout):
        card = RoundedCard()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)
        cl.addWidget(StyledLabel("📁 输入设置", font_size=12, bold=True, color=C["title"]))

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft)
        form.setContentsMargins(0, 0, 0, 0)

        form.addRow(StyledLabel("视频文件:", font_size=11, color=C["text_secondary"]), self._make_path_row("video"))
        form.addRow(StyledLabel("输出文件:", font_size=11, color=C["text_secondary"]), self._make_path_row("output", default="output.ply"))
        form.addRow(StyledLabel("工作目录:", font_size=11, color=C["text_secondary"]), self._make_path_row("workdir", default="./workdir"))

        cl.addLayout(form)
        parent_layout.addWidget(card)

    def _add_param_section(self, parent_layout):
        card = RoundedCard()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)
        cl.addWidget(StyledLabel("⚙️ 训练参数", font_size=12, bold=True, color=C["title"]))

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft)
        form.setContentsMargins(0, 0, 0, 0)

        # Sampling mode
        self.sampling_combo = StyledComboBox()
        self.sampling_combo.addItems(["均匀采样", "智能采样", "两阶段采样"])
        self.sampling_combo.setCurrentIndex(0)

        self.fps_spin = StyledDoubleSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(15)
        self.fps_spin.setSingleStep(1)

        self.scale_spin = StyledDoubleSpinBox()
        self.scale_spin.setRange(0.1, 1.0)
        self.scale_spin.setValue(0.5)
        self.scale_spin.setSingleStep(0.05)

        self.min_frames_spin = StyledSpinBox()
        self.min_frames_spin.setRange(10, 500)
        self.min_frames_spin.setValue(30)

        self.max_frames_spin = StyledSpinBox()
        self.max_frames_spin.setRange(10, 500)
        self.max_frames_spin.setValue(200)

        self.epochs_spin = StyledSpinBox()
        self.epochs_spin.setRange(10, 10000)
        self.epochs_spin.setValue(1000)
        self.epochs_spin.setSingleStep(100)

        self.eval_every_spin = StyledSpinBox()
        self.eval_every_spin.setRange(1, 1000)
        self.eval_every_spin.setValue(10)

        self.max_gaussians_spin = StyledSpinBox()
        self.max_gaussians_spin.setRange(50000, 2000000)
        self.max_gaussians_spin.setValue(300000)
        self.max_gaussians_spin.setSingleStep(50000)

        # SH
        self.sh_degree_combo = StyledComboBox()
        self.sh_degree_combo.addItems(["0 — 仅漫反射", "1 — 基础高光", "2 — 增强高光", "3 — 完整视角相关"])
        self.sh_degree_combo.setCurrentIndex(3)

        self.sh_warmup_spin = StyledSpinBox()
        self.sh_warmup_spin.setRange(0, 5000)
        self.sh_warmup_spin.setValue(1000)
        self.sh_warmup_spin.setSingleStep(100)

        self.ssim_warmup_spin = StyledSpinBox()
        self.ssim_warmup_spin.setRange(0, 5000)
        self.ssim_warmup_spin.setValue(500)
        self.ssim_warmup_spin.setSingleStep(100)

        self.ssim_weight_spin = StyledDoubleSpinBox()
        self.ssim_weight_spin.setRange(0, 1.0)
        self.ssim_weight_spin.setValue(0.2)
        self.ssim_weight_spin.setSingleStep(0.05)

        self.random_bg_cb = QCheckBox("随机黑白背景")
        self.random_bg_cb.setChecked(True)
        self.random_bg_cb.setStyleSheet(f"color: {C['text_primary']}; font-size: 11px; font-weight: 500;")

        self.train_focal_cb = QCheckBox("自动微调焦距")
        self.train_focal_cb.setChecked(True)
        self.train_focal_cb.setStyleSheet(f"color: {C['text_primary']}; font-size: 11px; font-weight: 500;")

        self.enable_k1_cb = QCheckBox("启用径向畸变校正 (k1)")
        self.enable_k1_cb.setChecked(False)
        self.enable_k1_cb.setStyleSheet(f"color: {C['text_primary']}; font-size: 11px; font-weight: 500;")

        # Device & estimator
        self.device_combo = StyledComboBox()
        if torch.cuda.is_available():
            self.device_combo.addItems(["自动", "CPU", "CUDA"])
            self._device_map = {"自动": "auto", "CPU": "cpu", "CUDA": "cuda"}
        else:
            self.device_combo.addItems(["自动", "CPU"])
            self._device_map = {"自动": "auto", "CPU": "cpu"}

        self.pose_estimator_combo = StyledComboBox()
        self.pose_estimator_combo.addItems(["ORB+EM", "COLMAP"])

        form.addRow(StyledLabel("采样模式:", font_size=11, color=C["text_secondary"]), self.sampling_combo)
        form.addRow(StyledLabel("采样帧率:", font_size=11, color=C["text_secondary"]), self.fps_spin)
        form.addRow(StyledLabel("画面缩放:", font_size=11, color=C["text_secondary"]), self.scale_spin)
        form.addRow(StyledLabel("最少帧数:", font_size=11, color=C["text_secondary"]), self.min_frames_spin)
        form.addRow(StyledLabel("最多帧数:", font_size=11, color=C["text_secondary"]), self.max_frames_spin)
        form.addRow(StyledLabel("训练轮次:", font_size=11, color=C["text_secondary"]), self.epochs_spin)
        form.addRow(StyledLabel("评估间隔:", font_size=11, color=C["text_secondary"]), self.eval_every_spin)
        form.addRow(StyledLabel("高斯上限:", font_size=11, color=C["text_secondary"]), self.max_gaussians_spin)
        form.addRow(StyledLabel("SH 阶数:", font_size=11, color=C["text_secondary"]), self.sh_degree_combo)
        form.addRow(StyledLabel("SH 升温步数:", font_size=11, color=C["text_secondary"]), self.sh_warmup_spin)
        form.addRow(StyledLabel("SSIM 升温步数:", font_size=11, color=C["text_secondary"]), self.ssim_warmup_spin)
        form.addRow(StyledLabel("SSIM 最大权重:", font_size=11, color=C["text_secondary"]), self.ssim_weight_spin)
        form.addRow(StyledLabel("动态背景:", font_size=11, color=C["text_secondary"]), self.random_bg_cb)
        form.addRow(StyledLabel("焦距自校准:", font_size=11, color=C["text_secondary"]), self.train_focal_cb)
        form.addRow(StyledLabel("径向畸变:", font_size=11, color=C["text_secondary"]), self.enable_k1_cb)
        form.addRow(StyledLabel("计算设备:", font_size=11, color=C["text_secondary"]), self.device_combo)
        form.addRow(StyledLabel("姿态估算:", font_size=11, color=C["text_secondary"]), self.pose_estimator_combo)

        cl.addLayout(form)
        parent_layout.addWidget(card)

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C["bg"]}; }}
            QWidget {{ background-color: transparent; color: {C["text_primary"]}; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; }}
            QScrollBar:vertical {{ background-color: {C["bg"]}; width: 8px; border: none; }}
            QScrollBar::handle:vertical {{ background-color: {C["border"]}; border-radius: 4px; min-height: 24px; }}
            QScrollBar::handle:vertical:hover {{ background-color: {C["text_secondary"]}; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ border: none; background: none; }}
            QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
            QSplitter::handle {{ background-color: {C["border"]}; }}
        """)

    def _switch_tab(self, idx):
        self.stacked.setCurrentIndex(idx)
        btns = [self.tab_btn_preview, self.tab_btn_logs, self.tab_btn_loss]
        for i, btn in enumerate(btns):
            btn.setObjectName("tabBtnActive" if i == idx else "tabBtnInactive")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _browse_file(self, kind, target_edit):
        if kind == "video":
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        elif kind == "output":
            path, _ = QFileDialog.getSaveFileName(self, "保存 PLY", "output.ply", "PLY 文件 (*.ply);;所有文件 (*)")
        else:
            path = QFileDialog.getExistingDirectory(self, "选择工作目录")
        if path:
            target_edit.setText(path)
            if kind == "video":
                self._show_single_preview(path)

    def _show_single_preview(self, vp):
        try:
            cap = cv2.VideoCapture(vp)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok:
                    self.preview_widget.set_image(frame)
                cap.release()
        except Exception:
            pass

    def _on_start(self):
        video = self.video_edit.text().strip()
        output = self.output_edit.text().strip()
        workdir = self.workdir_edit.text().strip() or "./workdir"

        if not video or not os.path.isfile(video):
            QMessageBox.warning(self, "提示", "请选择有效的视频文件。")
            return
        if not output:
            QMessageBox.warning(self, "提示", "请指定输出文件名。")
            return

        # Map sampling mode
        mode_map = {"均匀采样": "uniform", "智能采样": "smart", "两阶段采样": "two-stage"}
        sampling_mode = mode_map.get(self.sampling_combo.currentText(), "uniform")

        # Map SH degree
        sh_text = self.sh_degree_combo.currentText()
        sh_degree = int(sh_text[0]) if sh_text else 0

        # Map device
        device = self._device_map.get(self.device_combo.currentText(), "auto")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Map pose estimator
        pose_estimator = "colmap" if self.pose_estimator_combo.currentIndex() == 1 else "opencv"

        config = {
            "video": video,
            "output": output,
            "workdir": workdir,
            "sampling_mode": sampling_mode,
            "fps": self.fps_spin.value(),
            "scale": self.scale_spin.value(),
            "min_frames": self.min_frames_spin.value(),
            "max_frames": self.max_frames_spin.value(),
            "num_epochs": self.epochs_spin.value(),
            "eval_every": self.eval_every_spin.value(),
            "max_gaussians": self.max_gaussians_spin.value(),
            "sh_degree": sh_degree,
            "sh_warmup_steps": self.sh_warmup_spin.value(),
            "ssim_warmup_steps": self.ssim_warmup_spin.value(),
            "ssim_weight_max": self.ssim_weight_spin.value(),
            "random_background": self.random_bg_cb.isChecked(),
            "train_focal": self.train_focal_cb.isChecked(),
            "enable_k1": self.enable_k1_cb.isChecked(),
            "device": device,
            "pose_estimator": pose_estimator,
        }

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_dot.set_status("running")
        self.status_text.setText("运行中…")
        self.status_text.setStyleSheet(f"color: {C['success']}; font-size: 10pt;")
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备启动…")
        self.progress_label.setStyleSheet(f"color: {C['text_secondary']}; font-size: 10pt;")
        self.log_viewer.clear()

        self._log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self._log("  3D 高斯溅射重建 启动")
        self._log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self._log(f"📹  视频:      {config['video']}")
        self._log(f"💾  输出:      {config['output']}")
        self._log(f"🎯  采样模式:  {config['sampling_mode']}")
        self._log(f"⚙️   帧率:      {config['fps']} FPS")
        self._log(f"📐  缩放比例:  {config['scale']}")
        self._log(f"🎞️   帧数范围:  {config['min_frames']}–{config['max_frames']}")
        self._log(f"🔄  训练轮次:  {config['num_epochs']}")
        self._log(f"💻  计算设备:  {config['device']}")
        self._log(f"📂  工作目录:  {config['workdir']}")
        self._log(f"🎨 SH 阶数:   {config['sh_degree']}")
        self._log(f"🔥 SH 升温:   {config['sh_warmup_steps']} 步")
        self._log(f"📈 SSIM 升温: {config['ssim_warmup_steps']} 步")
        self._log(f"🎲 动态背景:  {'是' if config['random_background'] else '否'}")
        self._log(f"🔍 焦距自校准: {'是' if config['train_focal'] else '否'}")
        self._log(f"🔮 径向畸变:  {'是' if config['enable_k1'] else '否'}")
        self._log(f"🗺️  姿态估算:  {config['pose_estimator']}")

        self.worker = PipelineWorker(config)
        self.worker.log_signal.connect(self._append_log)
        self.worker.progress_signal.connect(self._update_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.frame_paths_signal.connect(self._show_preview_frames)
        self.worker.loss_signal.connect(self._on_loss_update)
        self.worker.start()

    def _on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._log("\n⏹ 正在停止…")

    def _on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if success:
            self.status_dot.set_status("done")
            self.status_text.setText("完成 ✓")
            self.status_text.setStyleSheet(f"color: {C['success']}; font-size: 10pt;")
            self.progress_bar.setValue(100)
            self.progress_label.setText("重建完成！")
            self.progress_label.setStyleSheet(f"color: {C['success']}; font-size: 10pt;")
            self._log(f"✅ {message}")
            self._log("🎉 3D 重建完成！请在输出目录查看 PLY 文件。")
        else:
            self.status_dot.set_status("error")
            self.status_text.setText("失败 ✗")
            self.status_text.setStyleSheet(f"color: {C['danger']}; font-size: 10pt;")
            self.progress_label.setText("重建失败")
            self.progress_label.setStyleSheet(f"color: {C['danger']}; font-size: 10pt;")
            self._log(f"❌ {message}")

    def _append_log(self, msg):
        self.log_viewer.append_log(msg)

    def _log(self, msg):
        self.log_viewer.append_log(msg)

    def _update_progress(self, pct, text):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(text)

    def _on_loss_update(self, data):
        self.loss_curve_page.update_from_data(data)

    def _show_preview_frames(self, frame_paths):
        # Clear old thumbnails
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, fp in enumerate(frame_paths[:24]):
            if i % 3 == 0:
                row = QHBoxLayout()
                row.setSpacing(8)
                row.setAlignment(Qt.AlignCenter)
                self.thumb_layout.addLayout(row)

            try:
                img = cv2.imread(fp)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pix = QPixmap.fromImage(QImage(img.tobytes(), img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888))
                pix = pix.scaled(180, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                lbl = QLabel()
                lbl.setPixmap(pix)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(f"background-color: {C['bg']}; border: 1px solid {C['border_light']}; border-radius: 6px;")

                card = QWidget()
                cl = QVBoxLayout(card)
                cl.setContentsMargins(2, 2, 2, 2)
                cl.setSpacing(2)
                cl.addWidget(lbl)

                fl = StyledLabel(f"帧 {i}", font_size=9, color=C["text_muted"])
                fl.setAlignment(Qt.AlignCenter)
                cl.addWidget(fl)

                card.setStyleSheet(f"background-color: {C['bg_card']}; border: 1px solid {C['border_light']}; border-radius: 8px;")
                row.addWidget(card)
            except Exception:
                pass

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


# ============================================================================
# Entry point
# ============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("3D Gaussian Splatting")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()