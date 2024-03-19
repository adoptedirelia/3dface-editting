import sys
sys.path.append("../../")

import pickle
import functools
import torch
from configs import paths_config, global_config

import legacy
import dnnlib
import training


def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def load_tuned_G(run_id, type, full_path=None):
    w_path_dir = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}'

    if full_path is None:
        new_G_path = f'{w_path_dir}/{paths_config.pti_results_keyword}/{run_id}/model_{run_id}.pt'
    else:
        new_G_path = full_path

    with open(new_G_path, 'rb') as f:
        new_G = torch.load(f).to(global_config.device).eval()
    new_G = new_G.float()
    toogle_grad(new_G, False)
    return new_G


def load_3dgan():
    with dnnlib.util.open_url(paths_config.eg3d_ffhq) as fp:
        old_G = legacy.load_network_pkl(fp)['G_ema'].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G

def load_stylegan2d():
    with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
        old_G = pickle.load(f)['G_ema'].to(global_config.device).eval()
        old_G = old_G.float()
    return old_G