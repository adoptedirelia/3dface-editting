import gradio as gr

def image_classifier(inp):
    return {'cat': 0.3, 'dog': 0.7}

demo = gr.Interface(fn=image_classifier, inputs="image", outputs="label")
demo.launch() 
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            text1 = gr.Textbox()
            text2 = gr.Textbox()
        with gr.Column(scale=4):
            btn1 = gr.Button("Button 1")
            btn2 = gr.Button("Button 2")