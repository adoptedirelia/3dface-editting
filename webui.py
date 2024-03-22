import gradio as gr
import os
import PIL
from PIL import Image
from inversion.configs import paths_config, global_config
import imageio

def edit_pose(image,row,column):
    
    print(f"row:{row}")
    print(f"column:{column}")

    pic_path = paths_config.web_pose
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
    print(os.getcwd())
    os.chdir('../')
    final_save_path = paths_config.web_pose_output

    final = Image.open(final_save_path+os.listdir(final_save_path)[0])

    return final

def edit_style(image,prompt):
    

    if os.path.exists('./inversion/embeddings/PTI/picture')==False:

        os.chdir('./preprocess')
        os.system('run.bat')
        print("预处理完成！")
        # python run_pti.py

        os.chdir('../inversion')
        os.system('python run_pti.py')
        os.system(f'python run_optimization.py --description "{prompt}"')

    else:
        os.chdir('./inversion')
        # print(os.getcwd())

        os.system(f'python run_optimization.py --description "{prompt}"')

    # final_save_path = ./out/picture.jpg
    # print(os.getcwd())
        
    final_save_path = paths_config.styleclip_output_dir
    final = Image.open(final_save_path+os.listdir(paths_config.styleclip_output_dir)[0])
    os.chdir('../')

    return final


def generate_video(image):

    if os.path.exists('./inversion/embeddings/PTI/picture')==False:
        # 运行run.bat的内容\

        os.chdir('./preprocess')
        os.system('run.bat')
        print("预处理完成！")
        # python run_pti.py

        os.chdir('../inversion')
        os.system('python run_pti.py')
        os.system(f'python gen_samples.py --outdir video_out --ppl picture')
        os.system('python concate_video.py')

    # python gen_pos.py --ppl picture --col column --row row --outdir out
    else:
        os.chdir('./inversion')
        # print(os.getcwd())

        os.system(f'python gen_samples.py --outdir video_out --ppl picture')
        os.system('python concate_video.py')


    # final_save_path = ./out/picture.jpg
    # print(os.getcwd())


    a = gr.Video('F:\mycode\\3dface-pose-editting\inversion\output_video.mp4')
    os.chdir('../')
    return a

with gr.Blocks(title="3D Face Editing") as demo:
    #用markdown语法编辑输出一段话
    gr.Markdown(
        """
        # 3D人脸姿势和风格编辑
        """
    )
    # 设置tab选项卡
    with gr.Tab("Pose Edit"):
        #Blocks特有组件，设置所有子组件按垂直排列
        #垂直排列是默认情况，不加也没关系
        with gr.Row():
            with gr.Column():
                pose_input = gr.Image()
                row = gr.Slider(-1,1,info="水平方向调整",label="row",value=0)
                column = gr.Slider(-1,1,info="垂直方向调整",label="column",value=0)
                pose_button = gr.Button("Generate Pose")
            pose_output = gr.Image()

    with gr.Tab("Style Edit"):
        #Blocks特有组件，设置所有子组件按水平排列
        with gr.Row():
            with gr.Column():
                style_input = gr.Image()
                prompt = gr.Text(info="输入需要改变的地方")
            style_output = gr.Image()
        style_button = gr.Button("Generate Style")
    #设置折叠内容
    with gr.Tab("Generate Video"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Image()
                video_button = gr.Button("Generate Video")
            video_output = gr.Video()

    with gr.Accordion("介绍"):
        gr.Markdown("不知道写啥")

    
    pose_button.click(edit_pose, inputs=[pose_input,row,column], outputs=pose_output)
    style_button.click(edit_style, inputs=[style_input,prompt], outputs=style_output)
    video_button.click(generate_video,inputs=video_input,outputs=video_output)

demo.launch()

