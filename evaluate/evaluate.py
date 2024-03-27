import numpy as np
import cv2
import lpips
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage import io

# 越低越好
def mse(image1, image2):
    # 确保图像尺寸相同
    print(image1.shape,image2.shape)
    
    assert image1.shape == image2.shape
    
    # 计算差的平方
    diff = (image1.astype(float) - image2.astype(float)) ** 2
    
    # 计算均方误差
    mse_value = np.mean(diff)
    
    return mse_value

# 越低越好
def lpips_distance(image1, image2):
    # 加载LPIPS模型

    model = lpips.LPIPS(net='alex')

    # 读取图像并将其转换为PyTorch张量

    image1_tensor = lpips.im2tensor(image1)
    image2_tensor = lpips.im2tensor(image2)

    # 计算LPIPS距离
    distance = model.forward(image1_tensor, image2_tensor)

    return distance.item()

# 越高越好
def ms_ssim(image1, image2):
    # 读取图像
    # 计算 MS-SSIM
    score, _ = ssim(image1, image2, full=True,channel_axis=2)
    
    return score


if __name__ == '__main__':

    ori = cv2.imread('./img/ori.jpg')
    res = cv2.imread('./img/res.png')
    #res = cv2.imread('./img/out_e4e.jpg')

    ori= cv2.resize(ori, (512, 512))
    res= cv2.resize(res, (512, 512))


# 计算MSE
    mse_value = mse(ori, res)

    print("MSE:", mse_value)


    lpips_distance_value = lpips_distance(ori, res)

    print("LPIPS Distance:", lpips_distance_value)

    ms_ssim_score = ms_ssim(ori, res)

    print("MS-SSIM Score:", ms_ssim_score)