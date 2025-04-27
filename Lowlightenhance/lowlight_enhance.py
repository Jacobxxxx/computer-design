import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from .data.data import *
#from .loss.losses import *
# from net.CIDNet import CIDNet
from .net.retinex import Retinex
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFileDialog, QMessageBox, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtCore import Qt, QSize

def load_model():
    #当前文件所在的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #预训练模型的路径
    weighth_path = os.path.join(current_dir, 'pretrained_models', 'lowlight_enhance.pth')

    model = Retinex().cuda()

    model.load_state_dict(torch.load(weighth_path, map_location='cuda'))
    model.eval()

    return model

def lowlight_enhance(input_image, model):
    # 转换为 RGB 格式
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    transform = ToTensor()
    image_tensor = transform(input_image).unsqueeze(0).cuda()

    # 模型推理
    with torch.no_grad():
        output, _ = model(image_tensor)

    # 处理输出
    output_image = output.squeeze(0).cpu()  # 移除批量维度并移回 CPU
    output_image = output_image.permute(1, 2, 0).numpy()  # 转为 [H, W, C]
    output_image = (output_image * 255).astype("uint8")  # 转为 uint8 格式

    # 转换为 QtGui.QImage
    h, w, _ = output_image.shape
    bytes_per_line = 3 * w
    qimg = QtGui.QImage(output_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

    return qimg


class LowLightEnhanceUI(QtWidgets.QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.enhanced_image = None
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
        self.setWindowTitle("低光增强 - 视驾通")
        self.setGeometry(200, 200, 1000, 600)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.setContentsMargins(20, 15, 20, 15)
        main_layout.setSpacing(20)

        # 图像显示区
        image_layout = QtWidgets.QHBoxLayout()
        image_layout.setSpacing(20)

        self.original_view = QtWidgets.QLabel("原始图像区域")
        self.original_view.setAlignment(QtCore.Qt.AlignCenter)
        self.original_view.setToolTip("低光输入图像")

        self.processed_view = QtWidgets.QLabel("处理后图像区域")
        self.processed_view.setAlignment(QtCore.Qt.AlignCenter)
        self.processed_view.setToolTip("低光增强结果")

        image_layout.addWidget(self.original_view)
        image_layout.addWidget(self.processed_view)

        # 操作按钮组
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(15)

        self.btn_select = QtWidgets.QPushButton(QtGui.QIcon("open_file.svg"), "选择文件")
        self.btn_select.setIconSize(QtCore.QSize(24, 24))
        self.btn_select.clicked.connect(self.select_file)

        self.btn_process = QtWidgets.QPushButton(QtGui.QIcon("start_process.svg"), "开始增强")
        self.btn_process.setObjectName("process_btn")
        self.btn_process.setIconSize(QtCore.QSize(24, 24))
        self.btn_process.clicked.connect(self.enhance_image)
        self.btn_process.setEnabled(False)

        self.btn_save = QtWidgets.QPushButton(QtGui.QIcon("save_file.svg"), "保存结果")
        self.btn_save.setIconSize(QtCore.QSize(24, 24))
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
            try:
                # 兼容中文路径
                self.original_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if self.original_image is None:
                    raise ValueError("无法读取图像文件")
                self.display_image(self.original_image, self.original_view)
                self.btn_process.setEnabled(True)
                self.btn_save.setEnabled(False)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "文件错误", f"加载失败:\n{str(e)}")

    def enhance_image(self):
        if hasattr(self, 'original_image'):
            self.btn_process.setEnabled(False)
            self.btn_process.setText("处理中...")
            QtWidgets.QApplication.processEvents()

            try:
                enhanced_image = lowlight_enhance(self.original_image, self.model)
                self.display_image(enhanced_image, self.processed_view)
                self.enhanced_image = enhanced_image
                self.btn_save.setEnabled(True)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "处理错误",
                                               f"增强失败:\n{str(e)}\n错误类型: {type(e).__name__}")
            finally:
                self.btn_process.setEnabled(True)
                self.btn_process.setText("开始增强")

    def display_image(self, image, label):
        try:
            if isinstance(image, QImage):
                pixmap = QtGui.QPixmap.fromImage(image).scaled(
                label.width() - 20, label.height() - 20,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            else:
                if image is None or image.size == 0:
                    raise ValueError("无效图像输入")

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # 内存连续性处理
                if not image_rgb.flags['C_CONTIGUOUS']:
                    image_rgb = np.ascontiguousarray(image_rgb)

                h, w, ch = image_rgb.shape
                img_bytes = image_rgb.tobytes()

                qimg = QtGui.QImage(img_bytes, w, h,
                                    ch * w, QtGui.QImage.Format_RGB888)
                pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
                    label.width() - 20, label.height() - 20,
                    QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            label.setPixmap(pixmap)

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "显示错误",
                                           f"图像显示失败: {str(e)}\n"
                                           f"尺寸: {image.shape if hasattr(image, 'shape') else 'N/A'}")

    def save_file(self):
        if hasattr(self, 'enhanced_image'):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_dir = os.path.join(current_dir, "output")
            os.makedirs(dataset_dir, exist_ok=True)
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存图像", "",
                "PNG Files (*.png);;JPEG Files (*.jpg)", options=options)

            if file_path:
                try:
                    self.enhanced_image.save(file_path)
                    QtWidgets.QMessageBox.information(self, "保存成功",
                                                      f"图像已保存至:\n{file_path}", QtWidgets.QMessageBox.Ok)
                except Exception as e:
                    QtWidgets.QMessageBox.critical(self, "保存失败",
                                                   f"保存时出错:\n{str(e)}")

    def resizeEvent(self, event):
        for label in [self.original_view, self.processed_view]:
            pixmap = label.pixmap()
            if pixmap:
                label.setPixmap(pixmap.scaled(
                    label.width() - 20, label.height() - 20,
                    QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        super().resizeEvent(event)
