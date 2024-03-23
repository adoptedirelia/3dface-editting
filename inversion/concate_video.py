import os
import re
import imageio
from configs import paths_config, global_config


def images_to_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # 排序文件名，确保按正确的顺序加载图片
    images.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    image_files = [os.path.join(image_folder, img) for img in images]
    writer = imageio.get_writer(video_name, fps=fps)

    for image_file in image_files:
        writer.append_data(imageio.imread(image_file))

    writer.close()

image_folder = paths_config.web_video_output.split('/')[-1]
video_name = 'output_video.mp4'
fps = 30

images_to_video(image_folder, video_name, fps)
