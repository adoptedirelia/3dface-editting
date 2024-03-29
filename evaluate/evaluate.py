import numpy as np
import cv2
import lpips
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os 


# 加载预训练的ResNet模型
resnet_model = models.resnet18(pretrained=True)
# 去掉最后一层全连接层
resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])
# 设置为评估模式
resnet_model.eval()

# 图像预处理函数
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_and_preprocess_image(img):
    image = Image.fromarray(img)
    #image = image.resize((512,512))
    image = preprocess(image)
    # 添加一个维度作为批处理维度
    image = image.unsqueeze(0)
    return image

# 提取图像的特征
def extract_features(image_path):
    image = load_and_preprocess_image(image_path)
    with torch.no_grad():
        features = resnet_model(image)
    return features.squeeze().numpy()

# 计算两张图像之间的ID相似度
# 越高越好
def id_similarity(image1, image2):
    features1 = extract_features(image1)
    features2 = extract_features(image2)
    # 计算余弦相似度
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity


# 越低越好
def mse(image1, image2):
    # 确保图像尺寸相同
    #print(image1.shape,image2.shape)
    
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


def show(dic):
    for i in dic:
        print(i)
        print(sum(dic[i])/len(dic[i]))

if __name__ == '__main__':
    
    lst = ['ours.jpg','out_e4e.jpg','out_SG2.jpg','out_SG2_p.jpg']

    dic_mse = {'ours':[],'out_e4e':[],'out_SG2':[],'out_SG2_p':[]}
    dic_ssim = {'ours':[],'out_e4e':[],'out_SG2':[],'out_SG2_p':[]}
    dic_lpips = {'ours':[],'out_e4e':[],'out_SG2':[],'out_SG2_p':[]}
    dic_id = {'ours':[],'out_e4e':[],'out_SG2':[],'out_SG2_p':[]}
    
    
    for title in os.listdir('./img'):
    
        if title == '.DS_Store':
            continue
        ori = cv2.imread(f'./img/{title}/origin.jpg')
        
        for i in lst:
            print("********************************")
            print(f"计算{i}的指标")

            res = cv2.imread(f'./img/{title}/{i}')
            #res = cv2.imread('./img/out_e4e.jpg')


        # 计算MSE
            mse_value = mse(ori, res)

            #print("MSE:", mse_value)


            lpips_distance_value = lpips_distance(ori, res)

            #print("LPIPS Distance:", lpips_distance_value)

            ms_ssim_score = ms_ssim(ori, res)

            #print("MS-SSIM Score:", ms_ssim_score)

            ID_value = id_similarity(ori,res)

            print(f"MSE: {mse_value}")
            print(f"LPIPS: {lpips_distance_value}")
            print(f"SSIM: {ms_ssim_score}")
            print(f"id similarity: {ID_value}")
            
            dic_mse[i.split('.')[0]].append(mse_value)
            dic_lpips[i.split('.')[0]].append(lpips_distance_value)
            dic_ssim[i.split('.')[0]].append(ms_ssim_score)
            dic_id[i.split('.')[0]].append(ID_value)

    print(dic_mse)
    print(dic_lpips)
    print(dic_ssim)
    print(dic_id)

    show(dic_mse)
    show(dic_lpips)
    show(dic_ssim)
    show(dic_id)