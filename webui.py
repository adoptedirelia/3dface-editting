import gradio as gr
import os
import PIL
from PIL import Image

def greet(image,row,column):
    
    print(f"row:{row}")
    print(f"column:{column}")

    pic_path = './preprocess/inputs'
    pic_name = 'picture.jpg'
    save_path = pic_path+'/'+pic_name

    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
        print("Folder created successfully.")
    else:
        print("Folder already exists.")

    # 删除preprocess/inputs下的内容

    pic = Image.fromarray(image)
    
    pic.save(save_path)
    # 运行run.bat的内容
    
    # python run_pti.py

    # python gen_pos.py --ppl picture --col column --row row --outdir out

    # save_path = ./out/picture.jpg

    final = Image.open(save_path)
    return final

iface = gr.Interface(fn=greet, inputs=["image",gr.Slider(-1, 1),gr.Slider(-1, 1)], outputs="image")
iface.launch()

