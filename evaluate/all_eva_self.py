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
from tqdm import tqdm

file = open('./output.json','w')
dic = {}
for method in os.listdir('./self_com'):
    dic[method] = {}
    for ppl in os.listdir(f'./self_com/{method}'):
        dic[method][ppl] = {}
        for filedir in os.listdir(f'./self_com/{method}/{ppl}'):
            print(f"\033[1;31;44m{method}/{ppl}/{filedir}\033[0m")
            dic[method][ppl][filedir] = {}


            path = f'./self_com/{method}/{ppl}'
            ori = cv2.imread(f'{path}/ori/ori.png')
            ori = cv2.resize(ori,(512,512))
            if filedir == 'ori':
                continue 

            temp = cv2.imread(f'./{path}/{filedir}/{filedir}.png')
            temp = cv2.resize(temp,(512,512))

            print(ori.shape,temp.shape)
            a = evaluate.id_similarity(ori,temp)
            b = evaluate.lpips_distance(ori,temp)

            # 执行命令并捕获输出
            
            command = ["pytorch-fid",f"./{path}/ori/",f"./{path}/{filedir}"]
            result = subprocess.run(command, text=True, capture_output=True)
            content = result.stdout.split('\n')[-2].split(':')[-1]
            c = float(content.strip())


            command = ["clip-score",f"./{path}/{filedir}",f"./clip_text/{filedir}"]
            result = subprocess.run(command, text=True, capture_output=True)
            print(result.stdout)
            content = result.stdout.split('\n')[-2].split(':')[-1]
            d = float(content.strip())
            e = evaluate.ms_ssim(ori,temp)
            dic[method][ppl][filedir]['id_similarity'] = str(a)
            dic[method][ppl][filedir]['lpips'] = str(b)
            dic[method][ppl][filedir]['fid'] = str(c)
            dic[method][ppl][filedir]['clip_score'] = str(d)
            dic[method][ppl][filedir]['ssim'] = str(e)

            print(a,b,c,d,e)

json.dump(dic,file,indent=4,ensure_ascii=False)
file.close()


