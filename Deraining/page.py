import sys
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from skimage import img_as_ubyte
import utils
from natsort import natsorted
from basicsr.models.archs.restormer_arch import Restormer
import yaml


# 加载Restormer模型
def load_restormer_model(weights_path):
    yaml_file = 'Options/Deraining_Restormer.yml'
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    if 'type' in x['network_g']:
        del x['network_g']['type']

    model_restoration = Restormer(**x['network_g'])

    checkpoint = torch.load(weights_path, weights_only=True)
    model_restoration.load_state_dict(checkpoint['params'])
    model_restoration.cuda()
    model_restoration = torch.nn.DataParallel(model_restoration)
    model_restoration.eval()

    return model_restoration


# 实现雨天增强功能
def rain_enhance(input_image, model_restoration):
    factor = 8
    img = np.float32(input_image) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    input_ = img.unsqueeze(0).cuda()

    # Padding in case images are not multiples of 8
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    # Restormer 进行去雨处理
    with torch.no_grad():
        restored = model_restoration(input_)

    # Unpad images to original dimensions
    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    return img_as_ubyte(restored)


class ImageEnhancerApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # Initialize model
        self.model_restoration = load_restormer_model('./pretrained_models/deraining.pth')

        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle('图像增强工具')
        self.setGeometry(100, 100, 900, 600)

        # Layout
        layout = QtWidgets.QVBoxLayout()

        # Top button layout
        button_layout = QtWidgets.QHBoxLayout()

        # Load image button
        self.load_button = QtWidgets.QPushButton('加载图像')
        self.load_button.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_button)

        # Enhance mode dropdown
        self.enhance_mode = QtWidgets.QComboBox()
        self.enhance_mode.addItems(["雨天增强", "低光增强", "雾天增强"])
        button_layout.addWidget(self.enhance_mode)

        # Enhance image button
        self.enhance_button = QtWidgets.QPushButton('增强图像')
        self.enhance_button.clicked.connect(self.enhance_image)
        button_layout.addWidget(self.enhance_button)

        # Save button
        self.save_button = QtWidgets.QPushButton('保存图像')
        self.save_button.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_button)

        layout.addLayout(button_layout)

        # Image display
        self.image_label = QtWidgets.QLabel('未加载图像')
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Set layout
        self.setLayout(layout)

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图像文件", "",
                                                             "Images (*.png *.xpm *.jpg *.jpeg)", options=options)
        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image)

    def enhance_image(self):
        if not hasattr(self, 'original_image'):
            QtWidgets.QMessageBox.warning(self, "警告", "请先加载图像")
            return

        mode = self.enhance_mode.currentText()

        if mode == "雨天增强":
            enhanced_image = rain_enhance(self.original_image, self.model_restoration)
        elif mode == "低光增强":
            enhanced_image = low_light_enhance(self.original_image)  # 需要定义
        elif mode == "雾天增强":
            enhanced_image = foggy_day_enhance(self.original_image)  # 需要定义

        self.display_image(enhanced_image)

    def display_image(self, image):
        # Convert to RGB and then to QPixmap for displaying in QLabel
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.image_label.setPixmap(pixmap.scaled(600, 400, QtCore.Qt.KeepAspectRatio))

    def save_image(self):
        if not hasattr(self, 'original_image'):
            QtWidgets.QMessageBox.warning(self, "警告", "请先加载图像")
            return

        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存图像", "", "Images (*.png *.jpg)",
                                                             options=options)
        if file_path:
            cv2.imwrite(file_path, self.original_image)
            QtWidgets.QMessageBox.information(self, "保存成功", "图像已保存")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = ImageEnhancerApp()
    ex.show()
    sys.exit(app.exec_())
