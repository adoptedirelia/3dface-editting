import gradio as gr
import os

def greet(image,row,column):
    print(f"row:{row}")
    print(f"column:{column}")

    return 0

iface = gr.Interface(fn=greet, inputs=["image",gr.Slider(-1, 1),gr.Slider(-1, 1)], outputs="image")
iface.launch()

