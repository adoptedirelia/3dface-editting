from PIL import Image 
import os 


if __name__ == '__main__':
    title = 'Foreign_male'
    directory = f'./img/{title}'
    if not os.path.exists(directory):
        try:
            # 创建目录
            os.makedirs(directory)
            print(f"目录 '{directory}' 创建成功")
        except OSError as e:
            # 打印错误信息
            print(f"创建目录 '{directory}' 失败: {e}")
    else:
        print(f"目录 '{directory}' 已存在")

    image1 = Image.open('./temp/ori.jpg')
    image1 = image1.resize((512,512))
    image1.save(f'{directory}/origin.jpg')


    image1 = Image.open('./temp/res.png')
    image1 = image1.resize((512,512))
    image1.save(f'{directory}/ours.jpg')


    image1 = Image.open('./temp/out_e4e.jpg')
    image1 = image1.resize((512,512))
    image1.save(f'{directory}/out_e4e.jpg')


    image1 = Image.open('./temp/out_SG2.jpg')
    image1 = image1.resize((512,512))
    image1.save(f'{directory}/out_SG2.jpg')


    image1 = Image.open('./temp/out_SG2_plus.jpg')
    image1 = image1.resize((512,512))
    image1.save(f'{directory}/out_SG2_p.jpg')
