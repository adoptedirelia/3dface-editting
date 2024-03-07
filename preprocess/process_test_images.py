"""
    Processes a directory containing *.jpg/png and outputs crops and poses.
"""
import glob
import os
import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='/media/data6/ericryanchan/mafu/Deep3DFaceRecon_pytorch/test_images')
parser.add_argument('--gpu', default=0)
args = parser.parse_args()

print('Processing images:', sorted(glob.glob(os.path.join(args.input_dir, "*"))))

# Compute facial landmarks.
print("Computing facial landmarks for model...")
cmd = "python batch_mtcnn.py"
input_flag = " --in_root " + args.input_dir
cmd += input_flag
print(cmd)
subprocess.run([cmd], shell=True)

# Run model inference to produce crops and raw poses.
print("Running model inference...")
cmd = "python test.py"
input_flag = " --img_folder=" + args.input_dir
gpu_flag = " --gpu_ids=" + str(args.gpu) 
model_name_flag = " --name=pretrained"
model_file_flag = " --epoch=20 "
cmd += input_flag + gpu_flag + model_name_flag + model_file_flag
print(cmd)
subprocess.run([cmd], shell=True)

# Process poses into our representation -- produces a cameras.json file.
print("Processing final poses...")
cmd = "python 3dface2idr.py"
input_flag = " --in_root " + os.path.join(args.input_dir, "epoch_20_000000")
cmd += input_flag
print(cmd)
subprocess.run([cmd], shell=True)

# Perform final cropping of 1024x1024 images.
print("Processing final crops...")
cmd = "python final_crop.py"
input_flag = " --in_root " + os.path.join(args.input_dir, "crop_1024")
cmd += input_flag
print(cmd)
subprocess.run([cmd], shell=True)

'''
python batch_mtcnn.py --in_root ./inputs
python test.py --img_folder=./inputs --gpu_ids=0 --name=pretrained --epoch=20
python 3dface2idr.py --in_root ./inputs/epoch_20_000000
python final_crop.py --in_root ./inputs/crop_1024
'''