import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from Dehaze.dehaze import load_dehaze_model, DehazeUI
from Deraining.deraining import load_restormer_model, RainEnhanceUI
from Lowlightenhance.lowlight_enhance import load_model, LowLightEnhanceUI
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize

class ImageEnhancerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model_restoration = load_restormer_model()
        self.dehaze_model = load_dehaze_model()
        self.low_light_model = load_model()
        self.initUI()
        self.init_style()

    def init_style(self):
        self.setStyleSheet("""
            QWidget {
                background: #0A192F;
                color: #FFFFFF;
                font-family: 'Microsoft YaHei';
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
            QLabel#title {
                font-size: 24px;
                color: #00E5FF;
                font-weight: bold;
            }
        """)

    def initUI(self):
        self.setWindowTitle("路视达 - 自动驾驶辅助系统")
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 左侧车辆展示区
        left_panel = QWidget()
        left_panel.setFixedWidth(400)
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("环境感知可视化"))
        # 车辆3D模型占位图
        car_image = QLabel()
        car_image.setPixmap(QPixmap("car_model.png").scaled(380, 300, Qt.KeepAspectRatio))
        left_layout.addWidget(car_image)
        left_panel.setLayout(left_layout)

        # 右侧功能操作区
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(50, 50, 50, 50)

        # 系统标题
        title = QLabel("恶劣天气增强系统")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignCenter)

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

        # 系统状态栏
        status_bar = QHBoxLayout()
        status_bar.addWidget(QLabel("GPU: 12%"))
        status_bar.addWidget(QLabel("显存: 54%"))
        status_bar.addStretch()
        status_bar.addWidget(QLabel("图像增强算法 V2.1"))

        right_layout.addWidget(title)
        right_layout.addLayout(btn_group)
        right_layout.addStretch()
        right_layout.addLayout(status_bar)

        right_panel.setLayout(right_layout)

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
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = ImageEnhancerApp()
    window.show()
    sys.exit(app.exec_())