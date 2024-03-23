import gradio as gr
import os
import PIL
from PIL import Image
from inversion.configs import paths_config, global_config
import imageio
import shutil

def preprocess(image):

    pic_path = paths_config.web_pose
    pic_name = 'picture.jpg'
    save_path = pic_path+'/'+pic_name


    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
        print("Folder created successfully.")
    else:
        shutil.rmtree(paths_config.web_pose)
        os.makedirs(pic_path)
        print("Folder already exists.")

    pic = Image.fromarray(image)
    pic.save(save_path)

    os.chdir('./preprocess')
    os.system('run.bat')
    print("预处理完成！")
    # python run_pti.py

    os.chdir('../inversion')
    os.system('python run_pti.py')
    os.chdir('../')
    final_path = paths_config.processed_pic
    final = Image.open(final_path)
    return final


def edit_pose(image,row,column):
    # edit_pose最终输出文件夹： './inversion/out/
    print(f"row:{row}")
    print(f"column:{column}")
    outdir = paths_config.web_pose_output.split('/')[-1]
    if os.path.exists('./inversion/embeddings/PTI/picture')==False:
        # 运行run.bat的内容\


        pic_path = paths_config.web_pose
        pic_name = 'picture.jpg'
        save_path = pic_path+'/'+pic_name
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
            print("Folder created successfully.")
        else:
            shutil.rmtree(paths_config.web_pose)
            os.makedirs(pic_path)
            print("Folder already exists.")

        pic = Image.fromarray(image)
        pic.save(save_path)
        os.chdir('./preprocess')
        os.system('run.bat')
        print("预处理完成！")
        # python run_pti.py

        os.chdir('../inversion')
        os.system('python run_pti.py')


        os.system(f'python gen_pos.py --ppl picture --col {column} --row {row} --outdir {outdir}')

    # python gen_pos.py --ppl picture --col column --row row --outdir out
    else:
        os.chdir('./inversion')
        # print(os.getcwd())

        os.system(f'python gen_pos.py --ppl picture --col {column} --row {row} --outdir {outdir}')
    # final_save_path = ./out/picture.jpg
    # print(os.getcwd())
    print(os.getcwd())
    os.chdir('../')
    final_save_path = paths_config.web_pose_output

    final = Image.open(final_save_path+'/'+os.listdir(final_save_path)[0])

    return final

def edit_style(image,prompt,step,lr,l2_lambda):
    # 最终输出文件夹: './inversion/styleclip_output'
    # 中间文件： './inversion/styleclip_temp'

    if os.path.exists('./inversion/embeddings/PTI/picture')==False:
        pic_path = paths_config.web_pose
        pic_name = 'picture.jpg'
        save_path = pic_path+'/'+pic_name
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
            print("Folder created successfully.")
        else:
            shutil.rmtree(paths_config.web_pose)
            os.makedirs(pic_path)
            print("Folder already exists.")

        pic = Image.fromarray(image)
        pic.save(save_path)
        os.chdir('./preprocess')
        os.system('run.bat')
        print("预处理完成！")
        # python run_pti.py

        os.chdir('../inversion')
        os.system('python run_pti.py') 
        os.system(f'python run_optimization.py --description "{prompt} --step {step} --lr {lr} --l2_lambda {l2_lambda}"')

    else:
        os.chdir('./inversion')
        # print(os.getcwd())

        os.system(f'python run_optimization.py --description "{prompt} --step {step} --lr {lr} --l2_lambda {l2_lambda}"')

    # final_save_path = ./out/picture.jpg
    # print(os.getcwd())
        
    final_save_path = paths_config.styleclip_output_dir
    final = Image.open(final_save_path+os.listdir(paths_config.styleclip_output_dir)[0])
    os.chdir('../')
    shutil.rmtree(paths_config.styleclip_temp)
    os.makedirs(paths_config.styleclip_temp)
    return final


def generate_video(image,fps,frames):
    # 最终输出文件： './inversion/output_video.mp4'
    # 中间文件: './inversion/video_out/'

    outdir = paths_config.web_video_output.split('/')[-1]

    if os.path.exists('./inversion/embeddings/PTI/picture')==False:
        pic_path = paths_config.web_pose
        pic_name = 'picture.jpg'
        save_path = pic_path+'/'+pic_name
        if not os.path.exists(pic_path):
            os.makedirs(pic_path)
            print("Folder created successfully.")
        else:
            shutil.rmtree(paths_config.web_pose)
            os.makedirs(pic_path)
            print("Folder already exists.")

        pic = Image.fromarray(image)
        pic.save(save_path)
        os.chdir('./preprocess')
        os.system('run.bat')
        print("预处理完成！")
        # python run_pti.py

        os.chdir('../inversion')
        os.system('python run_pti.py')
        os.system(f'python gen_samples.py --outdir {outdir} --ppl picture --frames {frames}')
        os.system(f'python concate_video.py --fps {fps}')

    # python gen_pos.py --ppl picture --col column --row row --outdir out
    else:
        os.chdir('./inversion')
        # print(os.getcwd())

        os.system(f'python gen_samples.py --outdir {outdir} --ppl picture --frames {frames}')
        os.system(f'python concate_video.py --fps {fps}')


    # final_save_path = ./out/picture.jpg
    # print(os.getcwd())

    
    a = gr.Video('F:\mycode\\3dface-pose-editting\inversion\output_video.mp4')
    os.chdir('../')
    shutil.rmtree(paths_config.web_video_output)
    os.makedirs(paths_config.web_video_output)
    return a

with gr.Blocks(title="3D Face Editing") as demo:
    #用markdown语法编辑输出一段话
    gr.Markdown(
        """
        # 3D人脸姿势和风格编辑
        """
    )
    # 设置tab选项卡
    with gr.Tab("Preprocess Picture"):
        with gr.Row():
            with gr.Column():
                raw_img = gr.Image()
                raw_button = gr.Button("Preprocess")
            processed_img = gr.Image()
        with gr.Accordion(gr.Markdown("# Preprocess info")):
            gr.Markdown("图像预处理阶段，需要消耗比较多的时间")

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
        with gr.Accordion(gr.Markdown("# Pose info")):
            gr.Markdown(
                """

                该阶段可以对输入的人物的方向进行调整测试，如输入正脸可以输出左脸/右脸/上方/下方
                参数：
                - row 
                对人物水平方向进行调整，负数为左侧观看（人物右脸），正数为右侧观看（人物左脸）
                - column 
                对人物垂直方向进行调整，负数为从下方观看，正数为从上方观看
                """)

    with gr.Tab("Style Edit"):
        #Blocks特有组件，设置所有子组件按水平排列
        with gr.Row():
            with gr.Column():
                style_input = gr.Image()
                prompt = gr.Text(info="输入需要改变的地方")
                step = gr.Slider(100,1000,value=300,label='step',info="训练轮数",step=1)
                lr = gr.Slider(0,1,value=0.1,label='lr',info="学习率")
                l2_lambda = gr.Slider(0,1,value=0.008,label='l2_lambda',info="")
                
                style_button = gr.Button("Generate Style")

            style_output = gr.Image()

        with gr.Accordion(gr.Markdown("# Style info")):
            gr.Markdown(
                """

                该阶段可以对输入的人物的风格进行调整测试，如a lady with blue hair等
                参数：
                - prompt
                输入人物的对应风格描述
                - step
                训练轮数，step越大效果越好
                - lr 
                学习率，可以进行调整获取最佳输出效果
                - l2_lambda 
                对输出风格不满意也可以对其进行适当调整
                """)
    #设置折叠内容
    with gr.Tab("Generate Video"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Image()
                fps = gr.Slider(24,120,value=60,label='fps',info="视频帧率")
                frames = gr.Slider(30,600,value=120,label='frames',info="视频总帧数",step=1)
                
                video_button = gr.Button("Generate Video")
                
            video_output = gr.Video()
        with gr.Accordion(gr.Markdown("# Video info")):
            gr.Markdown(
                """

                该阶段可以对输入的人物输出一个样例视频
                参数：
                - fps
                输出视频的帧率
                - frames
                输出视频的帧数
                """)


    raw_button.click(preprocess,inputs=raw_img,outputs=processed_img)
    pose_button.click(edit_pose, inputs=[pose_input,row,column], outputs=pose_output)
    style_button.click(edit_style, inputs=[style_input,prompt,step,lr,l2_lambda], outputs=style_output)
    video_button.click(generate_video,inputs=[video_input,fps,frames],outputs=video_output)

demo.launch()

