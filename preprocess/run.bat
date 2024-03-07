python batch_mtcnn.py --in_root ./inputs
python test.py --img_folder=./inputs --gpu_ids=0 --name=pretrained --epoch=20
python 3dface2idr.py --in_root ./inputs/epoch_20_000000
python final_crop.py --in_root ./inputs/crop_1024