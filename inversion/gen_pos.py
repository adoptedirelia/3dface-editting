# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import pickle
import cv2

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

#----------------------------------------------------------------------------

@click.command()
@click.option('--ppl', 'people', help='choose a picture', required=True)
@click.option('--col', 'col', help='column', required=True)
@click.option('--row', 'row', help='row', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    people: str,
    col:float,
    row:float
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    #print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    #with dnnlib.util.open_url(network_pkl) as f:
    #    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new

    os.makedirs(outdir, exist_ok=True)

    ppl = people
        
    with open(f'./embeddings/PTI/{ppl}/model_{ppl}.pt', 'rb') as f_new: 
        G = torch.load(f_new).cuda()

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    # Generate images.

    #print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    z = torch.from_numpy(np.random.RandomState(1234).randn(1, G.z_dim)).to(device)

    imgs = []
    angle_p = float(col)
    angle_y = float(row)

    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
    cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
    conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)
    conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

    # ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    with open(f'./embeddings/PTI/{ppl}/optimized_noise_dict.pickle', 'rb') as file:
        # 使用 pickle.load() 方法加载 pickle 文件中的对象
        data = pickle.load(file)
        ws = torch.tensor(data['projected_w']).cuda()
    print(camera_params.shape)
    img = G.synthesis(ws, camera_params)['image']
    

    '''
    img = img['image_depth'][0]
    img = -img
    img = (img - img.min()) / (img.max() - img.min()) * 2 - 1
    print(img.shape)
    img = (img*127.5+128).clamp(0,255).to(torch.uint8)[0]
    img = img.cpu().detach().numpy()
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('./show.jpg',img)
    '''
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    imgs.append(img)

    img = torch.cat(imgs, dim=2)

    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/{ppl}.png')

    if shapes:
        # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
        max_batch=1000000

        samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
        samples = samples.to(z.device)
        sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
        transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
        transformed_ray_directions_expanded[..., -1] = -1

        head = 0
        with tqdm(total = samples.shape[1]) as pbar:
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    
                    #sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                    # ws = G.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=False)
                    planes = G.backbone.synthesis(ws, update_emas=False, noise_mode='const')
                    planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
                    sigma = G.renderer.run_model(planes, G.decoder, samples[:, head:head+max_batch],  transformed_ray_directions_expanded[:, :samples.shape[1]-head], G.rendering_kwargs)['sigma']

                    sigmas[:, head:head+max_batch] = sigma
                    head += max_batch
                    pbar.update(max_batch)

        sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
        sigmas = np.flip(sigmas, 0)

        # Trim the border of the extracted cube
        pad = int(30 * shape_res / 256)
        pad_value = -1000
        sigmas[:pad] = pad_value
        sigmas[-pad:] = pad_value
        sigmas[:, :pad] = pad_value
        sigmas[:, -pad:] = pad_value
        sigmas[:, :, :pad] = pad_value
        sigmas[:, :, -pad:] = pad_value

        if shape_format == '.ply':
            from shape_utils import convert_sdf_samples_to_ply
            print(sigma.shape)

            convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'{ppl}.ply'), level=10)
        elif shape_format == '.mrc': # output mrc
            with mrcfile.new_mmap(os.path.join(outdir, f'{ppl}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
