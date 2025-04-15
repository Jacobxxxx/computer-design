import os
import torch
import torch.nn.functional as F
import numpy as np
from skimage import img_as_ubyte
from basicsr.models.archs.restormer_arch import Restormer
import yaml
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 使用相对路径构建yaml文件的路径
yaml_file = os.path.join(current_dir, 'Options', 'Deraining_Restormer.yml')

# 检查文件是否存在
if not os.path.exists(yaml_file):
    print(f"文件 {yaml_file} 不存在！")

# 加载Restormer模型
def load_restormer_model():
    # 预训练模型的路径
    weights_path = 'D:/ProgramData/pycharm/Image-Enhancement/Deraining/pretrained_models/deraining.pth'

    # 获取当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 使用相对路径构建yaml文件的路径
    yaml_file = os.path.join(current_dir, 'Options', 'Deraining_Restormer.yml')

    # 检查文件是否存在
    if not os.path.exists(yaml_file):
        print(f"文件 {yaml_file} 不存在！")
        return None

    # 尝试从yaml文件中加载配置
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    # 加载yaml文件，获取配置
    with open(yaml_file, 'r') as f:
        x = yaml.load(f, Loader=Loader)

    # 去除不需要的'type'字段
    if 'type' in x['network_g']:
        del x['network_g']['type']

    # 初始化 Restormer 模型
    model_restoration = Restormer(**x['network_g'])

    # 加载训练好的模型权重
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

class RainEnhanceUI(QtWidgets.QWidget):
    def __init__(self, model_restoration):
        super().__init__()

        self.model_restoration = model_restoration

        self.initUI()

    def initUI(self):
        self.setWindowTitle('雨天增强')
        self.setGeometry(100, 100, 800, 600)

        layout = QtWidgets.QVBoxLayout()

        # Left side: original image
        self.original_image_label = QtWidgets.QLabel('原始图像')
        self.original_image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.original_image_label)

        # Right side: processed image
        self.processed_image_label = QtWidgets.QLabel('处理后的图像')
        self.processed_image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.processed_image_label)

        # File selection button
        self.select_button = QtWidgets.QPushButton('选择文件')
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        # Process button
        self.process_button = QtWidgets.QPushButton('开始处理')
        self.process_button.clicked.connect(self.process_image)
        layout.addWidget(self.process_button)

        # Save button
        self.save_button = QtWidgets.QPushButton('保存文件')
        self.save_button.clicked.connect(self.save_file)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def select_file(self):
        options = QFileDialog.Options()
        self.file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.xpm *.jpg *.jpeg)",
                                                        options=options)
        if self.file_path:
            self.original_image = cv2.imread(self.file_path)
            self.display_image(self.original_image, self.original_image_label)

    def process_image(self):
        if hasattr(self, 'original_image'):
            enhanced_image = rain_enhance(self.original_image, self.model_restoration)
            self.display_image(enhanced_image, self.processed_image_label)

    def display_image(self, image, label):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(400, 300, QtCore.Qt.KeepAspectRatio))

    def save_file(self):
        if hasattr(self, 'original_image'):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "Images (*.png *.jpg)", options=options)
            if file_path:
                cv2.imwrite(file_path, self.original_image)
                QtWidgets.QMessageBox.information(self, "保存成功", "图像已保存")