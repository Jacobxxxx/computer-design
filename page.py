import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from Dehaze.dehaze import load_dehaze_model, DehazeUI
from Deraining.deraining import load_restormer_model, RainEnhanceUI
from Lowlightenhance.lowlight_enhance import load_model, LowLightEnhanceUI
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon, QPainter
from PyQt5.QtCore import Qt, QSize


class ImageEnhancerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model_restoration = load_restormer_model()
        self.dehaze_model = load_dehaze_model()
        self.low_light_model = load_model()
        self.initUI()
        self.init_style()
        self.background = QPixmap("background.png")  # 加载背景图片

    def init_style(self):
        self.setStyleSheet("""
            QWidget {
                color: #FFFFFF;
                font-family: 'Microsoft YaHei';
                background: transparent;
            }
            QPushButton {
                background: rgba(0,229,255,0.15);
                border: 2px solid #00E5FF;
                border-radius: 8px;
                padding: 15px 25px;
                font-size: 16px;
                min-width: 180px;
            }
            QPushButton:hover {
                background: rgba(0,229,255,0.3);
                border: 2px solid #00E5FF;
                box-shadow: 0 0 10px rgba(0,229,255,0.5);
            }
            QPushButton:pressed {
                background: rgba(0,229,255,0.5);
            }
            QLabel#main_title {
                font-size: 36px;
                color: #00E5FF;
                font-weight: bold;
                padding: 15px 0;
            }

        """)

    def paintEvent(self, event):
        """ 绘制背景图片 """
        painter = QPainter(self)
        if not self.background.isNull():
            scaled_bg = self.background.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            painter.drawPixmap(
                (self.width() - scaled_bg.width()) // 2,
                (self.height() - scaled_bg.height()) // 2,
                scaled_bg
            )

    def initUI(self):
        self.setWindowTitle("视驾通")
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ===================== 左侧面板 =====================
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_panel.setAttribute(Qt.WA_TranslucentBackground)
        left_layout = QVBoxLayout()

        # 主标题
        title = QLabel("视驾通")
        title.setObjectName("main_title")
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)



        left_panel.setLayout(left_layout)

        # ===================== 右侧面板 =====================
        right_panel = QWidget()
        right_panel.setAttribute(Qt.WA_TranslucentBackground)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(50, 50, 50, 50)

        # 功能按钮组
        btn_group = QVBoxLayout()
        btn_group.setSpacing(20)
        functions = [
            ("雨天增强", "rain_enhance.svg"),
            ("低光增强", "low_light.svg"),
            ("雾天增强", "fog_enhance.svg")
        ]

        for text, icon in functions:
            btn = QPushButton()
            btn.setIcon(QIcon(icon))
            btn.setIconSize(QSize(32, 32))
            btn.setText(text)
            btn.setToolTip(f"启动{text}功能模块")
            btn.clicked.connect(self.create_handler(text))
            btn_group.addWidget(btn)

        right_layout.addLayout(btn_group)
        right_layout.addStretch()
        right_panel.setLayout(right_layout)

        # 组合布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        self.setLayout(main_layout)

    def create_handler(self, func_name):
        def handler():
            if func_name == "雨天增强":
                self.show_rainy_ui()
            elif func_name == "低光增强":
                self.show_low_light_ui()
            elif func_name == "雾天增强":
                self.show_dehaze_ui()

        return handler

    def show_rainy_ui(self):
        self.rainy_ui = RainEnhanceUI(self.model_restoration)
        self.rainy_ui.show()

    def show_low_light_ui(self):
        self.low_light_ui = LowLightEnhanceUI(self.low_light_model)
        self.low_light_ui.show()

    def show_dehaze_ui(self):
        self.dehaze_ui = DehazeUI(self.dehaze_model)
        self.dehaze_ui.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 统一字体设置
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    # 高DPI支持
    app.setAttribute(Qt.AA_EnableHighDpiScaling)
    window = ImageEnhancerApp()
    window.show()
    sys.exit(app.exec_())