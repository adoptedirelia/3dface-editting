import os
import shutil

# 定义图片所在的目录
img_dir = './self_com'

for method in os.listdir(f'./{img_dir}'):
    #for ppl in os.listdir(f'./{img_dir}/{method}'):
# 遍历目录中的所有文件
    for filename in os.listdir(f'./{img_dir}/{method}'):
        if filename.endswith('.png'):  # 确保处理的是PNG图片文件
            # 创建与图片同名的文件夹路径
            folder_path = os.path.join(f'./{img_dir}/{method}', filename[:-4])
            # 创建文件夹
            os.makedirs(folder_path, exist_ok=True)
            # 构造图片的原始完整路径
            original_path = os.path.join(f'./{img_dir}/{method}', filename)
            # 构造图片的新路径
            new_path = os.path.join(folder_path, filename)
            # 移动图片
            shutil.move(original_path, new_path)

print("图片已成功移动到各自的文件夹。")
