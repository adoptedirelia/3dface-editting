import os
import sys
import json
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from IPython.display import display
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper
import torchvision.transforms as transforms


with open('./embeddings/PTI/zjm/optimized_noise_dict.pickle', 'rb') as file:
  # 使用 pickle.load() 方法加载 pickle 文件中的对象
  data = pickle.load(file)
  w_pivot = torch.tensor(data['projected_w']).cuda()


with open(f'./embeddings/PTI/zjm/model_zjm.pt', 'rb') as f_new: 
    new_G = torch.load(f_new).cuda()


if os.path.basename(paths_config.input_pose_path).split(".")[1] == "json":
    f = open(paths_config.input_pose_path)
    target_pose = np.asarray(json.load(f)[paths_config.input_id]['pose']).astype(np.float32)
    f.close()
    o = target_pose[0:3, 3]
    o = 2.7 * o / np.linalg.norm(o)
    target_pose[0:3, 3] = o
    target_pose = np.reshape(target_pose, -1)    
else:
    target_pose = np.load(paths_config.input_pose_path).astype(np.float32)
    target_pose = np.reshape(target_pose, -1)

intrinsics = np.asarray([4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0]).astype(np.float32)
target_pose = np.concatenate([target_pose, intrinsics])
target_pose = torch.tensor(target_pose, device=global_config.device).unsqueeze(0)

latent_editor = LatentEditorWrapper()
latents_after_edit = latent_editor.get_single_interface_gan_edits(w_pivot, [-1.5, 1.5])


for direction, factor_and_edit in latents_after_edit.items():
  i=0
  print(f'Showing {direction} change')
  for latent in factor_and_edit.values():
    i += 1
    new_image = new_G.synthesis(latent, target_pose, noise_mode='const', force_fp32 = True)
    new_image = new_image['image']
    img = (new_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).detach().cpu().numpy()[0] 
    print(img.shape)

    resized_image = Image.fromarray(img,mode='RGB').resize((256,256)) 

    resized_image.save(f'./editting_result/interface_result_{direction}_{i}.jpg')
