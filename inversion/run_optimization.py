import argparse
import math
import os
import pickle
import numpy as np
import torch
import torchvision
from torch import optim
from tqdm import tqdm
from configs import paths_config, global_config
import json
from models.StyleCLIP.criteria.clip_loss import CLIPLoss

import clip
#from StyleCLIP.utils import ensure_checkpoint_exists


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def main(args):
    # ensure_checkpoint_exists(args.ckpt)
    text_inputs = torch.cat([clip.tokenize(args.description)]).cuda()
    os.makedirs(args.results_dir, exist_ok=True)
    new_generator_path = f'F:\mycode\\3dface-pose-editting\inversion\embeddings\PTI\picture\model_picture.pt'


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

    with open(new_generator_path, 'rb') as f:
        G = torch.load(f).cuda().eval()

    args.latent_path = "F:\mycode\\3dface-pose-editting\inversion\embeddings\PTI\picture\optimized_noise_dict.pickle"
    if args.latent_path:
        with open(args.latent_path, 'rb') as file:
            # 使用 pickle.load() 方法加载 pickle 文件中的对象
            data = pickle.load(file)
            latent_code_init = torch.tensor(data['projected_w']).cuda()
    elif args.mode == "edit":
        latent_code_init_not_trunc = torch.randn(1, 512).cuda()
        with torch.no_grad():
            latent_code_init = G.mapping(latent_code_init_not_trunc, None)

    latent = latent_code_init.detach().clone()
    latent.requires_grad = True
    with torch.no_grad():
        img_orig = G.synthesis(latent, target_pose, noise_mode='const', force_fp32=True)['image']
    clip_loss = CLIPLoss(args)

    optimizer = optim.Adam([latent], lr=args.lr)

    pbar = tqdm(range(args.step))

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr

        img_gen = G.synthesis(latent, target_pose, noise_mode='const', force_fp32=True)['image']
        #print(text_inputs.shape)
        #print(img_gen.shape)
        c_loss = clip_loss(img_gen, text_inputs)

        if args.mode == "edit":
            l2_loss = ((latent_code_init - latent) ** 2).sum()
            loss = c_loss + args.l2_lambda * l2_loss
        else:
            loss = c_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(
            (
                f"loss: {loss.item():.4f};"
            )
        )
        if args.save_intermediate_image_every > 0 and i % args.save_intermediate_image_every == 0:
            with torch.no_grad():
                img_gen = G.synthesis(latent, target_pose, noise_mode='const', force_fp32=True)['image']

            torchvision.utils.save_image(img_gen,
                                         f"./out/{str(i).zfill(5)}.png",
                                         normalize=True, value_range=(-1, 1))

    if args.mode == "edit":

        final_result = torch.cat([img_orig, img_gen])
    else:
        final_result = img_gen

    return final_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--description", type=str, default="a smiling face",
                        help="the text that guides the editing/generation")
    parser.add_argument("--stylegan_size", type=int, default=512, help="StyleGAN resolution")
    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--step", type=int, default=500, help="number of optimization steps")
    parser.add_argument("--mode", type=str, default="edit", choices=["edit", "free_generation"],
                        help="choose between edit an image an generate a free one")
    parser.add_argument("--l2_lambda", type=float, default=0.008,
                        help="weight of the latent distance (used for editing only)")
    parser.add_argument("--latent_path", type=str, default=None,
                        help="starts the optimization from the given latent code if provided. Otherwose, starts from"
                             "the mean latent in a free generation, and from a random one in editing. "
                             "Expects a .pt format")
    parser.add_argument("--truncation", type=float, default=0.7,
                        help="used only for the initial latent vector, and only when a latent code path is"
                             "not provided")
    parser.add_argument("--save_intermediate_image_every", type=int, default=20,
                        help="if > 0 then saves intermidate results during the optimization")
    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()

    result_image = main(args)

    torchvision.utils.save_image(result_image.detach().cpu(), os.path.join(args.results_dir, "final_result.jpg"),
                                 normalize=True, scale_each=True, value_range=(-1, 1))
