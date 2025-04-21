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
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize
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
class DehazeUI(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.processed_image = None
        self.initUI()
        self.init_style()

    def init_style(self):
        self.setStyleSheet("""
            QWidget {
                background: #0A192F;
                color: #FFFFFF;
                font-family: 'Microsoft YaHei';
            }
            QLabel {
                border: 2px solid #00E5FF;
                border-radius: 8px;
                min-width: 400px;
                min-height: 300px;
            }
            QPushButton {
                background: rgba(0,229,255,0.15);
                border: 2px solid #00E5FF;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background: rgba(0,229,255,0.3);
                box-shadow: 0 0 8px rgba(0,229,255,0.5);
            }
            QPushButton#process_btn {
                background: rgba(255,0,85,0.15);
                border-color: #FF0055;
            }
            QPushButton#process_btn:hover {
                background: rgba(255,0,85,0.3);
                box-shadow: 0 0 8px rgba(255,0,85,0.5);
            }
        """)

    def initUI(self):
        self.setWindowTitle("去雾增强 - 路视达")
        self.setGeometry(200, 200, 1000, 600)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)



        # 图像显示区
        image_layout = QHBoxLayout()
        image_layout.setSpacing(20)

        self.original_view = QLabel("原始图像区域")
        self.original_view.setAlignment(Qt.AlignCenter)
        self.original_view.setToolTip("原始输入图像")

        self.processed_view = QLabel("处理后图像区域")
        self.processed_view.setAlignment(Qt.AlignCenter)
        self.processed_view.setToolTip("去雾增强结果")

        image_layout.addWidget(self.original_view)
        image_layout.addWidget(self.processed_view)

        # 操作按钮组
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.btn_select = QPushButton(QIcon("open_file.svg"), "选择文件")
        self.btn_select.setIconSize(QSize(24, 24))
        self.btn_select.clicked.connect(self.select_file)

        self.btn_process = QPushButton(QIcon("start_process.svg"), "开始处理")
        self.btn_process.setObjectName("process_btn")
        self.btn_process.setIconSize(QSize(24, 24))
        self.btn_process.clicked.connect(self.process_image)

        self.btn_save = QPushButton(QIcon("save_file.svg"), "保存结果")
        self.btn_save.setIconSize(QSize(24, 24))
        self.btn_save.clicked.connect(self.save_file)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_process)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addStretch()

        main_layout.addLayout(image_layout)
        main_layout.addLayout(btn_layout)
        self.setLayout(main_layout)

    def select_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", "",
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp)", options=options)

        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("无法读取图像文件")
                self.display_image(self.original_image, self.original_view)
            except Exception as e:
                QMessageBox.critical(self, "文件错误", f"加载图像失败:\n{str(e)}")

    def process_image(self):
        if hasattr(self, 'original_image'):
            self.btn_process.setEnabled(False)
            self.btn_process.setText("处理中...")
            QtWidgets.QApplication.processEvents()

            try:
                enhanced_image = dehaze_enhance(self.original_image, self.model)
                self.display_image(enhanced_image, self.processed_view)
                self.enhanced_image = enhanced_image
            except Exception as e:
                QMessageBox.critical(self, "处理错误",
                                     f"图像处理失败:\n{str(e)}\n错误类型: {type(e).__name__}")
            finally:
                self.btn_process.setEnabled(True)
                self.btn_process.setText("开始处理")

    def display_image(self, image, label):
        try:
            if image is None or image.size == 0:
                raise ValueError("无效的输入图像")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if not image_rgb.flags['C_CONTIGUOUS']:
                image_rgb = np.ascontiguousarray(image_rgb)

            h, w, ch = image_rgb.shape
            img_bytes = image_rgb.tobytes()

            qimg = QImage(img_bytes, w, h,
                          ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                label.width() - 20, label.height() - 20,
                Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)

        except Exception as e:
            QMessageBox.critical(self, "显示错误",
                                 f"无法显示图像: {str(e)}\n"
                                 f"尺寸: {image.shape if hasattr(image, 'shape') else 'N/A'}")

    def save_file(self):
        if hasattr(self, 'enhanced_image'):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "",
                "PNG Files (*.png);;JPEG Files (*.jpg)", options=options)

            if file_path:
                try:
                    cv2.imwrite(file_path,
                                cv2.cvtColor(self.enhanced_image, cv2.COLOR_RGB2BGR))
                    QMessageBox.information(self, "保存成功",
                                            f"图像已保存至:\n{file_path}", QMessageBox.Ok)
                except Exception as e:
                    QMessageBox.critical(self, "保存失败",
                                         f"保存文件时出错:\n{str(e)}")

    def resizeEvent(self, event):
        for label in [self.original_view, self.processed_view]:
            pixmap = label.pixmap()
            if pixmap:
                label.setPixmap(pixmap.scaled(
                    label.width() - 20, label.height() - 20,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)

