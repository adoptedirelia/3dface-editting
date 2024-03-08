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

    pic = Image.fromarray(image)
    
    pic.save(save_path)
    
    final = Image.open(save_path)
    return final

iface = gr.Interface(fn=greet, inputs=["image",gr.Slider(-1, 1),gr.Slider(-1, 1)], outputs="image")
iface.launch()

