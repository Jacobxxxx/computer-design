import torch
import torch.nn.functional as F
import numpy as np
from skimage import img_as_ubyte
from basicsr.models.archs.restormer_arch import Restormer
import yaml
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize

import os


current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_file = os.path.join(current_dir, 'Options', 'Deraining_Restormer.yml')


if not os.path.exists(yaml_file):
    print(f"文件 {yaml_file} 不存在！")

def load_restormer_model():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(current_dir, "pretrained_models","deraining.pth")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_file = os.path.join(current_dir, 'Options', 'Deraining_Restormer.yml')
    if not os.path.exists(yaml_file):
        print(f"文件 {yaml_file} 不存在！")
        return None
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(yaml_file, 'r') as f:
        x = yaml.load(f, Loader=Loader)

    if 'type' in x['network_g']:
        del x['network_g']['type']
    model_restoration = Restormer(**x['network_g'])

    checkpoint = torch.load(weights_path, weights_only=True)
    model_restoration.load_state_dict(checkpoint['params'])
    model_restoration.cuda()
    model_restoration = torch.nn.DataParallel(model_restoration)
    model_restoration.eval()

    return model_restoration

def rain_enhance(input_image, model_restoration):
    factor = 8
    img = np.float32(input_image) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    input_ = img.unsqueeze(0).cuda()

    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        restored = model_restoration(input_)

    restored = restored[:, :, :h, :w]
    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    return img_as_ubyte(restored)


class RainEnhanceUI(QWidget):
    def __init__(self, model_restoration):
        super().__init__()
        self.model_restoration = model_restoration
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
            
             /* 消息弹窗全局样式 */
        QMessageBox {
                background-color: #0A192F;
                font-family: 'Microsoft YaHei';
            }
    
            /   * 消息文字 */
        QMessageBox QLabel#qt_msgbox_label {
                color: #FFFFFF;
                font-size: 14px;
                qproperty-alignment: AlignCenter;
            }
    
            /* 按钮区域 */
        QMessageBox QDialogButtonBox {
                button-layout: WinLayout;  /* 按钮排列方式 */
            }
    
            /* 按钮样式 */
        QMessageBox QPushButton {
                background: rgba(0,229,255,0.15);
                border: 2px solid #00E5FF;
                color: #FFFFFF;
                min-width: 80px;
                padding: 8px 16px;
                border-radius: 5px;
            }
    
        QMessageBox QPushButton:hover {
                background: rgba(0,229,255,0.3);
                box-shadow: 0 0 8px rgba(0,229,255,0.5);
            }
        """)

    def initUI(self):
        self.setWindowTitle("雨天增强 - 视驾通")
        self.setGeometry(200, 200, 1000, 600)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)

        image_layout = QHBoxLayout()
        image_layout.setSpacing(20)

        self.original_view = QLabel("原始图像区域")
        self.original_view.setAlignment(Qt.AlignCenter)
        self.original_view.setToolTip("原始输入图像")

        self.processed_view = QLabel("处理后图像区域")
        self.processed_view.setAlignment(Qt.AlignCenter)
        self.processed_view.setToolTip("去雨增强结果")

        image_layout.addWidget(self.original_view)
        image_layout.addWidget(self.processed_view)


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
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_dir = os.path.join(current_dir, "test")
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像文件", directory=dataset_dir,
            filter="Images (*.png *.jpg *.jpeg *.bmp)", options=options)

        if file_path:
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.original_view)

    def process_image(self):
        if hasattr(self, 'original_image'):
            self.btn_process.setText("处理中...")
            QtWidgets.QApplication.processEvents()

            try:
                enhanced_image = rain_enhance(self.original_image, self.model_restoration)
                self.display_image(enhanced_image, self.processed_view)
                self.enhanced_image = enhanced_image
            except Exception as e:
                QMessageBox.critical(self, "处理错误", str(e))
            finally:
                self.btn_process.setText("开始处理")

    def display_image(self, image, label):
        try:

            if image is None or image.size == 0:
                raise ValueError("无效的输入图像")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if not image_rgb.flags['C_CONTIGUOUS']:
                image_rgb = np.ascontiguousarray(image_rgb)
            h, w, ch = image_rgb.shape
            bytes_per_line = ch * w

            img_bytes = image_rgb.tobytes()

            qimg = QImage(
                img_bytes,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            )

            pixmap = QPixmap.fromImage(qimg)

            scaled_pixmap = pixmap.scaled(
                label.width() - 20,
                label.height() - 20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            label.setPixmap(scaled_pixmap)

        except Exception as e:
            QMessageBox.critical(self, "图像显示错误",
                                 f"无法显示图像：\n{str(e)}\n"
                                 f"图像类型：{type(image)}\n"
                                 f"图像形状：{image.shape if hasattr(image, 'shape') else 'N/A'}")

    def save_file(self):
        if hasattr(self, 'enhanced_image'):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_dir = os.path.join(current_dir, "output")
            os.makedirs(dataset_dir, exist_ok=True)
            options = QFileDialog.Options()

            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", dataset_dir,
                "PNG Files (*.png);;JPEG Files (*.jpg)", options=options)
            if file_path:

                filename = os.path.basename(file_path)
                save_path = os.path.join(dataset_dir, filename)

                cv2.imwrite(save_path, cv2.cvtColor(self.enhanced_image, cv2.COLOR_RGB2BGR))
                QMessageBox.information(self, "保存成功",
                                        f"图像已保存至：\n{save_path}", QMessageBox.Ok)

    def resizeEvent(self, event):

        for label in [self.original_view, self.processed_view]:
            pixmap = label.pixmap()
            if pixmap:
                label.setPixmap(pixmap.scaled(
                    label.width() - 20, label.height() - 20,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation))
        super().resizeEvent(event)