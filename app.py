import gradio as gr
import os
import openai
from auto_backgrounds import generate_backgrounds, fake_generator, generate_draft
from utils.file_operations import hash_name

# note: App白屏bug：允许第三方cookie
# todo:
#   4. add auto_polishing function
#   5. Use some simple method for simple tasks (including: writing abstract, conclusion, generate keywords, generate figures...)
#       5.1 Use GPT 3.5 for abstract, conclusion, ... (or may not)
#       5.2 Use local LLM to generate keywords, figures, ...
#       5.3 Use embedding to find most related papers (find a paper dataset)
#       5.4 Use Semantic Scholar API instead of Arxiv API.
#   6. get logs when the procedure is not completed.
#   7. 自己的文件库； 更多的prompts
#   8. Change prompts to langchain

openai_key = os.getenv("OPENAI_API_KEY")
access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
if access_key_id is None or secret_access_key is None:
    print("Access keys are not provided. Outputs cannot be saved to AWS Cloud Storage.\n")
    IS_CACHE_AVAILABLE = False
else:
    IS_CACHE_AVAILABLE = True

if openai_key is None:
    print("OPENAI_API_KEY is not found in environment variables. The output may not be generated.\n")
    IS_OPENAI_API_KEY_AVAILABLE = False
else:
    openai.api_key = openai_key
    try:
        openai.Model.list()
        IS_OPENAI_API_KEY_AVAILABLE = True
    except Exception as e:
        IS_OPENAI_API_KEY_AVAILABLE = False



def clear_inputs(text1, text2):
    return "", ""


def wrapped_generator(title, description, openai_key = None,
                      template = "ICLR2022",
                      cache_mode = IS_CACHE_AVAILABLE, generator=None):
    # if `cache_mode` is True, then follow the following steps:
    #        check if "title"+"description" have been generated before
    #        if so, download from the cloud storage, return it
    #        if not, generate the result.
    if generator is None:
        # todo: add a Dropdown to select which generator to use.
        # generator = generate_backgrounds
        generator = generate_draft
        # generator = fake_generator
    if openai_key is not None:
        openai.api_key = openai_key
        openai.Model.list()

    if cache_mode:
        from utils.storage import list_all_files, download_file, upload_file
        # check if "title"+"description" have been generated before

        input_dict = {"title": title, "description": description, "generator": "generate_draft"} #todo: modify here also
        file_name = hash_name(input_dict) + ".zip"
        file_list = list_all_files()
        # print(f"{file_name} will be generated. Check the file list {file_list}")
        if file_name in file_list:
            # download from the cloud storage, return it
            download_file(file_name)
            return file_name
        else:
            # generate the result.
            # output = fake_generate_backgrounds(title, description, openai_key) # todo: use `generator` to control which function to use.
            output = generator(title, description,  template, "gpt-4")
            upload_file(output)
            return output
    else:
        # output = fake_generate_backgrounds(title, description, openai_key)
        output = generator(title, description,  template, "gpt-4")
        return output


theme = gr.themes.Monochrome(font=gr.themes.GoogleFont("Questrial")).set(
    background_fill_primary='#E5E4E2',
    background_fill_secondary = '#F6F6F6',
    button_primary_background_fill="#281A39"
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown('''
    # Auto-Draft: 文献整理辅助工具
    
    本Demo提供对[Auto-Draft](https://github.com/CCCBora/auto-draft)的auto_draft功能的测试。通过输入想要生成的论文名称（比如Playing atari with deep reinforcement learning)，即可由AI辅助生成论文模板.    
    
    ***2023-05-03 Update***: 在公开版本中为大家提供了输入OpenAI API Key的地址, 如果有GPT-4的API KEY的话可以在这里体验! 
    在这个Huggingface Organization里也提供一定额度的免费体验： [AUTO-ACADEMIC](https://huggingface.co/organizations/auto-academic/share/HPjgazDSlkwLNCWKiAiZoYtXaJIatkWDYM).
    如果有更多想法和建议欢迎加入QQ群里交流, 如果我在Space里更新了Key我会第一时间通知大家. 群号: ***249738228***.  
    
    ## 用法
    
    输入想要生成的论文名称（比如Playing Atari with Deep Reinforcement Learning), 点击Submit, 等待大概十分钟, 下载.zip格式的输出，在Overleaf上编译浏览.  
    ''')
    with gr.Row():
        with gr.Column(scale=2):
            key =  gr.Textbox(value=openai_key, lines=1, max_lines=1, label="OpenAI Key", visible=not IS_OPENAI_API_KEY_AVAILABLE)
            # generator = gr.Dropdown(choices=["学术论文", "文献总结"], value="文献总结", label="Selection", info="目前支持生成'学术论文'和'文献总结'.", interactive=True)
            title = gr.Textbox(value="Playing Atari with Deep Reinforcement Learning", lines=1, max_lines=1, label="Title", info="论文标题")
            description = gr.Textbox(lines=5, label="Description (Optional)", visible=False)

            with gr.Row():
                clear_button = gr.Button("Clear")
                submit_button = gr.Button("Submit")
        with gr.Column(scale=1):
            style_mapping = {True: "color:white;background-color:green", False: "color:white;background-color:red"} #todo: to match website's style
            availability_mapping = {True: "AVAILABLE", False: "NOT AVAILABLE"}
            gr.Markdown(f'''## Huggingface Space Status  
             当`OpenAI API`显示AVAILABLE的时候这个Space可以直接使用.    
             当`OpenAI API`显示NOT AVAILABLE的时候这个Space可以通过在左侧输入OPENAI KEY来使用. 需要有GPT-4的API权限. 
             当`Cache`显示AVAILABLE的时候, 所有的输入和输出会被备份到我的云储存中. 显示NOT AVAILABLE的时候不影响实际使用. 
            `OpenAI API`: <span style="{style_mapping[IS_OPENAI_API_KEY_AVAILABLE]}">{availability_mapping[IS_OPENAI_API_KEY_AVAILABLE]}</span>.  `Cache`: <span style="{style_mapping[IS_CACHE_AVAILABLE]}">{availability_mapping[IS_CACHE_AVAILABLE]}</span>.''')
            file_output = gr.File(label="Output")

    clear_button.click(fn=clear_inputs, inputs=[title, description], outputs=[title, description])
    submit_button.click(fn=wrapped_generator, inputs=[title, description, key], outputs=file_output)

demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch()
