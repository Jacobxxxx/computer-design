import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from .data.data import *
#from .loss.losses import *
# from net.CIDNet import CIDNet
from .net.retinex import Retinex
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog


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
        self.initUI()

    def initUI(self):
        self.setWindowTitle('低光增强')
        self.setGeometry(100, 100, 800, 600)

        layout = QtWidgets.QVBoxLayout()

        # 原始图像显示
        self.original_image_label = QtWidgets.QLabel('原始图像')
        self.original_image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.original_image_label)

        # 增强后图像显示
        self.enhanced_image_label = QtWidgets.QLabel('增强后图像')
        self.enhanced_image_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.enhanced_image_label)

        # 文件选择按钮
        self.select_button = QtWidgets.QPushButton('选择文件')
        self.select_button.clicked.connect(self.select_file)
        layout.addWidget(self.select_button)

        # 增强按钮
        self.enhance_button = QtWidgets.QPushButton('开始增强')
        self.enhance_button.clicked.connect(self.enhance_image)
        layout.addWidget(self.enhance_button)

        # 保存按钮
        self.save_button = QtWidgets.QPushButton('保存文件')
        self.save_button.clicked.connect(self.save_file)
        layout.addWidget(self.save_button)

        self.setLayout(layout)

    def select_file(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if file_path:
            self.file_path = file_path
            self.original_image = cv2.imread(file_path)
            self.display_image(self.original_image, self.original_image_label)
            self.enhance_button.setEnabled(True)  # 启用“开始增强”按钮
    
    def enhance_image(self):
        if hasattr(self, 'original_image'):
            enhanced_image = lowlight_enhance(self.original_image, self.model)
            self.enhanced_image_label.setPixmap(QtGui.QPixmap.fromImage(enhanced_image).scaled(400, 300, QtCore.Qt.KeepAspectRatio))

    def display_image(self, image, label):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image_rgb.shape
        bytes_per_line = 3 * w
        qimg = QtGui.QImage(image_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(400, 300, QtCore.Qt.KeepAspectRatio))

    def save_file(self):
        if hasattr(self, 'enhanced_image_label'):
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", "", "Images (*.png *.jpg)", options=options)
            if file_path:
                # 保存增强后的图像
                enhanced_image_np = self.enhanced_image_label.pixmap().toImage()
                enhanced_image_np.save(file_path)
                QtWidgets.QMessageBox.information(self, "保存成功", "增强后的图像已保存")
        else:
            QtWidgets.QMessageBox.warning(self, "保存失败", "请先处理图像后再保存")

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)

    model = load_model()
    low_light_enhance_ui = LowLightEnhanceUI(model)
    low_light_enhance_ui.show()

    sys.exit(app.exec_())