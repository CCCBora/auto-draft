import gradio as gr
import openai
from auto_backgrounds import generate_backgrounds

# todo: 　5. Add more functions in this demo.

def clear_inputs(text1, text2):
    return ("", "")

def wrapped_generate_backgrounds(title, description):
    if title == "Deep Reinforcement Learning":
        return "output.zip"
    else:
        return generate_backgrounds(title, description)


with gr.Blocks() as demo:
    gr.Markdown('''
    # Auto-Draft: 文献整理辅助工具
    
    本Demo提供对[Auto-Draft](https://github.com/CCCBora/auto-draft)的auto_backgrounds功能的测试。通过输入一个领域的名称（比如Deep Reinforcement Learning)，
    即可自动对这个领域的相关文献进行归纳总结. 
    
    *生成一篇论文需要15000 Tokens(大约0.5到0.8美元). 请用户酌情体验.*
    
    ## 用法
    
    输入一个领域的名称（比如Deep Reinforcement Learning), 点击Submit, 等待大概十分钟, 下载output.zip，在Overleaf上编译浏览.  
    ''')
    with gr.Row():
        with gr.Column():
            title = gr.Textbox(value="Deep Reinforcement Learning", lines=1, max_lines=1, label="Title")
            description = gr.Textbox(lines=5, label="Description (Optional)")

            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit")
        with gr.Column():
            file_output = gr.File()

    clear_button.click(fn=clear_inputs, inputs=[title, description], outputs=[title, description])
    submit_button.click(fn=wrapped_generate_backgrounds, inputs=[title, description], outputs=file_output)

demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch()