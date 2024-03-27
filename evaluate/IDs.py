import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

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

# 加载并处理图像
def load_and_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((512,512))
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
def id_similarity(image1_path, image2_path):
    features1 = extract_features(image1_path)
    features2 = extract_features(image2_path)
    # 计算余弦相似度
    similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return similarity

# 图片路径
image1_path = './img/ori.jpg'
image2_path = './img/out_e4e.jpg'

# 计算ID相似度
similarity_score = id_similarity(image1_path, image2_path)
print("ID Similarity Score:", similarity_score)
