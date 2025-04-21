import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore
from Dehaze.dehaze import load_dehaze_model,DehazeUI
from Deraining.deraining import load_restormer_model,RainEnhanceUI
from Lowlightenhance.lowlight_enhance import load_model,LowLightEnhanceUI
class ImageEnhancerApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Initialize model
        self.model_restoration = load_restormer_model()
        self.dehaze_model=load_dehaze_model()
        self.low_light_model=load_model()
        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle('图像增强工具')
        self.setGeometry(100, 100, 1000, 600)
        # Set background image
        self.set_background_image('D:/ProgramData/pycharm/Image-Enhancement/background.png')  # 替换为实际的图片路径

        # Layout
        layout = QtWidgets.QVBoxLayout()

        # Button layout for mode selection
        button_layout = QtWidgets.QHBoxLayout()

        # Add mode buttons (e.g., Rain Enhancement, Low Light, etc.)
        self.rainy_button = QtWidgets.QPushButton('雨天增强')
        self.rainy_button.clicked.connect(self.show_rainy_ui)
        button_layout.addWidget(self.rainy_button)

        self.low_light_button = QtWidgets.QPushButton('低光增强')
        self.low_light_button.clicked.connect(self.show_low_light_ui)
        button_layout.addWidget(self.low_light_button)

        self.foggy_button = QtWidgets.QPushButton('雾天增强')
        self.foggy_button.clicked.connect(self.show_dehaze_ui)
        button_layout.addWidget(self.foggy_button)

        layout.addLayout(button_layout)

        # Main display area (image display)
        self.image_display = QtWidgets.QLabel('欢迎使用图像增强工具')
        self.image_display.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.image_display)

        # Set layout
        self.setLayout(layout)

    def set_background_image(self, image_path):
        """ 设置窗口的背景图像 """
        self.setStyleSheet(f"background-image: url('{image_path}'); background-repeat: no-repeat; background-position: center;")

    def show_rainy_ui(self):
        self.rainy_ui = RainEnhanceUI(self.model_restoration)
        self.rainy_ui.show()

    def show_low_light_ui(self):
        self.low_light_ui = LowLightEnhanceUI(self.low_light_model)
        self.low_light_ui.show()

    def show_dehaze_ui(self):
        self.dehaze_ui=DehazeUI(self.dehaze_model)
        self.dehaze_ui.show()





if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = ImageEnhancerApp()
    ex.show()
    sys.exit(app.exec_())