import numpy as np
import os
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
from skimage import img_as_ubyte
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from basicsr.models.archs.restormer_arch import Restormer

# 初始化Tkinter，不需要创建窗口
Tk().withdraw()

# 通过文件对话框选择图片文件
def choose_file():
    filename = askopenfilename(title="Select an image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    return filename

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')

# 图片路径和输出路径
parser.add_argument('--weights', default='./pretrained_models/deraining.pth', type=str, help='Path to weights')

args = parser.parse_args()

# 选择需要测试的图片文件
input_image_path = choose_file()

# 输出路径，可以根据需要修改为特定的路径
output_image_path = './output_image.png'

# 加载Restormer模型
yaml_file = 'Options/Deraining_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')

model_restoration = Restormer(**x['network_g'])
checkpoint = torch.load(args.weights, weights_only=True)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# 加载输入图片
img = np.float32(utils.load_img(input_image_path))/255.  # 加载图片并归一化
img = torch.from_numpy(img).permute(2, 0, 1)
input_ = img.unsqueeze(0).cuda()

# Padding in case images are not multiples of 8
factor = 8
h, w = input_.shape[2], input_.shape[3]
H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
padh = H - h if h % factor != 0 else 0
padw = W - w if w % factor != 0 else 0
input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

# 恢复图片
with torch.no_grad():
    restored = model_restoration(input_)

# 去除Padding并恢复原始尺寸
restored = restored[:, :, :h, :w]

# 转换为0-255范围并保存
restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

# 保存处理后的图片
utils.save_img(output_image_path, img_as_ubyte(restored))
print(f"Processed image saved to: {output_image_path}")
