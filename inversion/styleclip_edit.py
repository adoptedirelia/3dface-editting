import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch
from configs import paths_config, hyperparameters, global_config
from utils.align_data import pre_process_images
from scripts.run_pti import run_PTI
from IPython.display import display
import matplotlib.pyplot as plt
from scripts.latent_editor_wrapper import LatentEditorWrapper
import torchvision.transforms as transforms
from scripts.pti_styleclip import styleclip_edit


styleclip_edit(use_multi_id_G=False, run_id='XMFUVSWULGRN', edit_types = ['angry'], use_wandb=False)


