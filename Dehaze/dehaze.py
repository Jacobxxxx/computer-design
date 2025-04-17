import warnings
import os
import torch.nn.functional as F
from collections import OrderedDict
from PyQt5 import QtWidgets, QtGui, QtCore
from Dehaze.models import *
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import torch
import numpy as np
import os
from skimage import img_as_ubyte
import cv2
from Dehaze.utils import write_img, chw_to_hwc
# 抑制timm库的FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")

# 抑制meshgrid相关警告
warnings.filterwarnings("ignore", category=UserWarning,
                        message="torch.meshgrid: in an upcoming release")


def load_model_weights(weight_path):
    """安全加载模型权重"""
    # 使用相对路径或动态计算的绝对路径
    #weight_path = os.path.abspath(weight_path)  # 转换为绝对路径
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"[权重文件缺失] 未找到预训练权重文件: {weight_path}")

    try:
        # 添加weights_only参数避免警告
        state_dict = torch.load(
            weight_path,
            weights_only = True
        )['state_dict']
    except TypeError:  # 处理旧版本PyTorch兼容性
        state_dict = torch.load(weight_path, map_location='cpu')['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):  # 处理可能的分布式训练权重
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def load_dehaze_model():
    """稳健的模型加载函数"""
    # 设备检测
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # 动态加载模型类
        model_class = eval('dehazeformer_b')
    except NameError:
        raise RuntimeError("""[模型加载错误] 未找到 'dehazeformer_b' 类定义，请确认：
        1. models.py文件中正确定义了该模型类
        2. 类名拼写与定义完全一致
        3. 已正确导入模型模块
        """)

    # 实例化模型
    model = model_class()
    model.cuda()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path=os.path.join(current_dir, 'saved_models', 'dehazeformer-b.pth')
  # 使用动态计算的绝对路径
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"[权重文件缺失] 未找到预训练权重文件: {weight_path}")

    try:
        model.load_state_dict(load_model_weights(weight_path))
    except Exception as e:
        raise RuntimeError(f"""
        [权重加载失败] 错误信息: {str(e)}
        可能原因：
        1. 模型结构与权重不匹配
        2. 文件损坏
        3. PyTorch版本不兼容
        解决方案：
        - 检查模型类定义是否与权重匹配
        - 重新下载权重文件
        - 更新PyTorch到最新版本
        """)

    return model.to(device).eval()


def dehaze_enhance(input_image, network):
    # 确保网络处于评估模式
    network.eval()

    # 预处理流程（与PairLoader完全一致）
    img = np.float32(input_image) / 255.0  # 原始代码的read_img隐含归一化
    img = img * 2 - 1  # 匹配训练时的[-1,1]标准化

    # 维度转换（保持与hwc_to_chw一致）
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()  # HWC->CHW

    # 动态填充（兼容任意尺寸输入）
    factor = 8  # 根据实际模型结构修改
    _, _, h, w = img_tensor.shape
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    if padh + padw > 0:
        img_tensor = F.pad(img_tensor, (0, padw, 0, padh), mode='reflect')

    # 模型推理（严格对齐test代码）
    with torch.no_grad():
        output = network(img_tensor).clamp_(-1, 1)  # 显式clamp

    # 后处理流程（与test代码的保存逻辑一致）
    output = output * 0.5 + 0.5  # [-1,1]->[0,1]转换
    output = output[:, :, :h, :w]  # 移除填充

    # 转换为numpy格式（使用chw_to_hwc逆操作）
    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # CHW->HWC

    return img_as_ubyte(output_np)


# 去雾功能GUI界面
class DehazeUI(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.processed_image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('去雾增强')  # 修正了标题
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
            enhanced_image = dehaze_enhance(self.original_image, self.model)
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

