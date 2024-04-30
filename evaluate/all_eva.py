import evaluate
import cv2
from PIL import Image
import numpy as np 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import os 
import subprocess
import pytorch_fid.fid_score
import json 

file = open('./output.json','w')
dic = {}
for method in os.listdir('./img'):
    print(method)
    dic[method] = {}
    for ppl in os.listdir(f'./img/{method}'):
        dic[method][ppl] = {}
        for filedir in os.listdir(f'./img/{method}/{ppl}'):
            dic[method][ppl][filedir] = {}


            path = f'./img/{method}/{ppl}'
            ori = cv2.imread(f'{path}/ori/ori.png')
            if filedir == 'ori':
                continue 

            temp = cv2.imread(f'./{path}/{filedir}/{filedir}.png')
            a = evaluate.id_similarity(ori,temp)
            b = evaluate.lpips_distance(ori,temp)

            # 执行命令并捕获输出
            '''            
            command = ["pytorch-fid",f"./{path}/ori/",f"./{path}/{filedir}"]
            result = subprocess.run(command, text=True, capture_output=True)

            content = result.stdout.split('\n')[-2].split(':')[-1]
            c = float(content.strip())


            command = ["clip-score",f"./{path}/{filedir}",f"./clip_text/{filedir}"]
            result = subprocess.run(command, text=True, capture_output=True)
            
            content = result.stdout.split('\n')[-2].split(':')[-1]
            d = float(content.strip())
            '''
            c = 0
            d = 0
            dic[method][ppl][filedir]['id_similarity'] = str(a)
            dic[method][ppl][filedir]['lpips'] = str(b)
            dic[method][ppl][filedir]['fid'] = str(c)
            dic[method][ppl][filedir]['clip_score'] = str(d)

            print(a,b,c,d)

json.dump(dic,file,indent=4,ensure_ascii=False)
file.close()


import os
import shutil

# 定义图片所在的目录
img_dir = './img'

for method in os.listdir(f'./{img_dir}'):
    for ppl in os.listdir(f'./{img_dir}/{method}'):
# 遍历目录中的所有文件
        for filename in os.listdir(f'./{img_dir}/{method}/{ppl}'):
            if filename.endswith('.jpg'):  # 确保处理的是PNG图片文件
                # 创建与图片同名的文件夹路径
                folder_path = os.path.join(img_dir, filename[:-4])
                # 创建文件夹
                os.makedirs(folder_path, exist_ok=True)
                # 构造图片的原始完整路径
                original_path = os.path.join(img_dir, filename)
                # 构造图片的新路径
                new_path = os.path.join(folder_path, filename)
                # 移动图片
                shutil.move(original_path, new_path)

print("图片已成功移动到各自的文件夹。")
