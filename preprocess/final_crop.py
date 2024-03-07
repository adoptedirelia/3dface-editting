import numpy as np
import os
import torch
import json
import argparse
import glob
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
parser.add_argument('--center_crop_size', type=int, default=700)
parser.add_argument('--output_size', type=int, default=512)
args = parser.parse_args()
in_root = args.in_root

images = glob.glob(os.path.join(in_root, "*"))

for img in images:
    im = Image.open(img)
    left = int(im.size[0]/2 - args.center_crop_size/2)
    upper = int(im.size[1]/2 - args.center_crop_size/2)
    right = left + args.center_crop_size
    lower = upper + args.center_crop_size
    im_cropped = im.crop((left, upper, right,lower))
    im_cropped = im_cropped.resize((args.output_size, args.output_size), resample=Image.LANCZOS)
    im_cropped.save(os.path.join(in_root, os.path.basename(img)))
