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

    if os.path.exists('./inversion/embeddings/PTI/picture')==False:
        # 运行run.bat的内容\

        os.chdir('./preprocess')
        os.system('run.bat')
        print("预处理完成！")
        # python run_pti.py

        os.chdir('../inversion')
        os.system('python run_pti.py')
        os.system(f'python gen_pos.py --ppl picture --col {column} --row {row} --outdir out')

    # python gen_pos.py --ppl picture --col column --row row --outdir out
    else:
        os.chdir('./inversion')
        # print(os.getcwd())

        os.system(f'python gen_pos.py --ppl picture --col {column} --row {row} --outdir out')
    # final_save_path = ./out/picture.jpg
    # print(os.getcwd())
    final_save_path = './out/picture.png'
    final = Image.open(final_save_path)
    os.chdir('../')
    return final

if __name__ == '__main__':
    iface = gr.Interface(fn=greet, inputs=["image",gr.Slider(-1, 1),gr.Slider(-1, 1)], outputs="image")
    iface.launch()

