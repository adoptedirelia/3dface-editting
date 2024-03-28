import cv2
import numpy as np
from PIL import Image
import os 
# 读取图像

for title in os.listdir('./img'):



    image1 = cv2.imread(f'./img/{title}/origin.jpg')
    image2 = cv2.imread(f'./img/{title}/ours.jpg')
    image3 = cv2.imread(f'./img/{title}/out_e4e.jpg')
    image4 = cv2.imread(f'./img/{title}/out_SG2.jpg')
    image5 = cv2.imread(f'./img/{title}/out_SG2_p.jpg')

    # 转换图像格式为RGB（Pillow要求的格式）
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image3_rgb = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)
    image4_rgb = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)
    image5_rgb = cv2.cvtColor(image5, cv2.COLOR_BGR2RGB)

    # 将OpenCV图像转换为Pillow Image对象
    pil_image1 = Image.fromarray(image1_rgb)
    pil_image2 = Image.fromarray(image2_rgb)
    pil_image3 = Image.fromarray(image3_rgb)
    pil_image4 = Image.fromarray(image4_rgb)
    pil_image5 = Image.fromarray(image5_rgb)

    # 获取图像的尺寸
    width1, height1 = pil_image1.size
    width2, height2 = pil_image2.size
    width3, height3 = pil_image3.size
    width4, height4 = pil_image4.size
    width5, height5 = pil_image5.size

    # 计算拼接后的图像的宽度和高度
    new_width = width1 + width2 + width3 + width4 + width5
    new_height = max(height1, height2, height3, height4, height5)

    # 创建一个新的画布，宽度为所有图像宽度之和，高度为所有图像中最高的高度
    new_image = Image.new('RGB', (new_width, new_height))

    # 将图像粘贴到新图像中
    new_image.paste(pil_image1, (0, 0))
    new_image.paste(pil_image2, (width1, 0))
    new_image.paste(pil_image3, (width1 + width2, 0))
    new_image.paste(pil_image4, (width1 + width2 + width3, 0))
    new_image.paste(pil_image5, (width1 + width2 + width3 + width4, 0))

    # 保存拼接后的图像
    new_image.save(f'./img/{title}/final.jpg')

