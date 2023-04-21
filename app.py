import gradio as gr
import openai
from auto_backgrounds import generate_backgrounds

# todo:
#       3. create a huggingface space. test it using multiple devices!
#       4. further polish auto_backgrounds.py. Make backgrounds have multiple subsection.
#       5. Design a good layout of huggingface space.

def clear_inputs(text1, text2):
    return ("", "")

with gr.Blocks() as demo:
    gr.Markdown('''
    # Auto-Draft: 论文结构辅助工具
    
    用法: 输入任意论文标题, 点击Submit, 等待大概十分钟, 下载output.zip.  
    ''')
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(value="Playing Atari Game with Deep Reinforcement Learning", lines=1, max_lines=1, label="Title")
            description = gr.Textbox(lines=5, label="Description (Optional)")

            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit")
        with gr.Column():
            file_output = gr.outputs.File()

    clear_button.click(fn=clear_inputs, inputs=[title, description], outputs=[title, description])
    submit_button.click(fn=generate_backgrounds, inputs=[title, description], outputs=file_output)

demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch()