import gradio as gr
import os
import openai
from auto_backgrounds import generate_backgrounds, fake_generator, generate_draft
from utils.file_operations import hash_name

# note: App白屏bug：允许第三方cookie
# todo:
#   6. get logs when the procedure is not completed. *
#   7. 自己的文件库； 更多的prompts
#   8. Decide on how to generate the main part of a paper * (Langchain/AutoGPT
#   9. Load .bibtex file to generate a pre-defined references list. *
#   1. 把paper改成纯JSON?
#   2. 实现别的功能
#   3. Check API Key GPT-4 Support.
#   8. Re-build some components using `langchain`
#           - in `references.py`, use PromptTemplates.format -> str
#           - in `gpt_interation`, use LLM
# future:
#   4. add auto_polishing function
#   12. Change link to more appealing color # after the website is built;
#   1. Check if there are any duplicated citations
#   2. Remove potential thebibliography and bibitem in .tex file

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


def wrapped_generator(paper_title, paper_description, openai_api_key=None,
                      template="ICLR2022",
                      cache_mode=IS_CACHE_AVAILABLE, generator=None):
    # if `cache_mode` is True, then follow the following steps:
    #        check if "title"+"description" have been generated before
    #        if so, download from the cloud storage, return it
    #        if not, generate the result.
    if generator is None:
        # todo: add a Dropdown to select which generator to use.
        # generator = generate_backgrounds
        generator = generate_draft
        # generator = fake_generator
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        openai.Model.list()

    if cache_mode:
        from utils.storage import list_all_files, download_file, upload_file
        # check if "title"+"description" have been generated before

        input_dict = {"title": paper_title, "description": paper_description,
                      "generator": "generate_draft"}  # todo: modify here also
        file_name = hash_name(input_dict) + ".zip"
        file_list = list_all_files()
        # print(f"{file_name} will be generated. Check the file list {file_list}")
        if file_name in file_list:
            # download from the cloud storage, return it
            download_file(file_name)
            return file_name
        else:
            # generate the result.
            # output = fake_generate_backgrounds(title, description, openai_key)
            # todo: use `generator` to control which function to use.
            output = generator(paper_title, paper_description, template, "gpt-4")
            upload_file(output)
            return output
    else:
        # output = fake_generate_backgrounds(title, description, openai_key)
        output = generator(paper_title, paper_description, template, "gpt-4")
        return output


theme = gr.themes.Default(font=gr.themes.GoogleFont("Questrial"))
# .set(
#     background_fill_primary='#E5E4E2',
#     background_fill_secondary = '#F6F6F6',
#     button_primary_background_fill="#281A39"
# )

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
            key = gr.Textbox(value=openai_key, lines=1, max_lines=1, label="OpenAI Key",
                             visible=not IS_OPENAI_API_KEY_AVAILABLE)

            # generator = gr.Dropdown(choices=["学术论文", "文献总结"], value="文献总结",
            # label="Selection", info="目前支持生成'学术论文'和'文献总结'.", interactive=True)

            # 每个功能做一个tab
            with gr.Tab("学术论文"):
                title = gr.Textbox(value="Playing Atari with Deep Reinforcement Learning", lines=1, max_lines=1,
                                   label="Title", info="论文标题")

                with gr.Accordion("高级设置", open=False):
                    description_pp = gr.Textbox(lines=5, label="Description (Optional)", visible=True,
                                                info="对希望生成的论文的一些描述. 包括这篇论文的创新点, 主要贡献, 等.")

                    interactive = False
                    gr.Markdown('''
                    ## 下面的功能我只做了UI, 还没来得及实现功能. 
                    ''')
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown('''
                            Upload .bib file (Optional)
                            
                            通过上传.bib文件来控制GPT-4模型必须参考哪些文献. 
                            ''')
                            bibtex_file = gr.File(label="Upload .bib file", file_types=["text"],
                                                  interactive=interactive)
                        with gr.Column():
                            search_engine = gr.Dropdown(label="Search Engine",
                                                        choices=["ArXiv", "Semantic Scholar", "Google Scholar", "None"],
                                                        value= "Semantic Scholar",
                                                        interactive=interactive,
                                                        info="用于决定GPT-4用什么搜索引擎来搜索文献. 选择None的时候仅参考给定文献.")
                            tldr = gr.Checkbox(value=True, label="TLDR;",
                                               info="选择此筐表示将使用Semantic Scholar的TLDR作为文献的总结.",
                                               interactive = interactive),
                            use_cache = gr.Checkbox(label="总是重新生成",
                                                    info="选择此筐表示将不会读取已经生成好的文章.",
                                               interactive = interactive)
                            slider = gr.Slider(minimum=1, maximum=30, value=20, label="最大参考文献数目",
                                               info="过多参考文献会超出Token数限制导致报错，这里限制最大参考文献数目.")

                with gr.Row():
                    clear_button_pp = gr.Button("Clear")
                    submit_button_pp = gr.Button("Submit", variant="primary")

            with gr.Tab("文献综述"):
                gr.Markdown('''
                <h1  style="text-align: center;">Coming soon!</h1>
                ''')
                # topic = gr.Textbox(value="Deep Reinforcement Learning", lines=1, max_lines=1,
                #                    label="Topic", info="文献主题")
                # with gr.Accordion("Advanced Setting"):
                #     description_lr = gr.Textbox(lines=5, label="Description (Optional)", visible=True,
                #                              info="对希望生成的综述的一些描述. 包括这篇论文的创新点, 主要贡献, 等.")
                # with gr.Row():
                #     clear_button_lr = gr.Button("Clear")
                #     submit_button_lr = gr.Button("Submit", variant="primary")
            with gr.Tab("论文润色"):
                gr.Markdown('''
                <h1  style="text-align: center;">Coming soon!</h1>
                ''')
            with gr.Tab("帮我想想该写什么论文!"):
                gr.Markdown('''
                <h1  style="text-align: center;">Coming soon!</h1>
                ''')

        with gr.Column(scale=1):
            style_mapping = {True: "color:white;background-color:green",
                             False: "color:white;background-color:red"}  # todo: to match website's style
            availability_mapping = {True: "AVAILABLE", False: "NOT AVAILABLE"}
            gr.Markdown(f'''## Huggingface Space Status  
             当`OpenAI API`显示AVAILABLE的时候这个Space可以直接使用.    
             当`OpenAI API`显示NOT AVAILABLE的时候这个Space可以通过在左侧输入OPENAI KEY来使用. 需要有GPT-4的API权限. 
             当`Cache`显示AVAILABLE的时候, 所有的输入和输出会被备份到我的云储存中. 显示NOT AVAILABLE的时候不影响实际使用. 
            `OpenAI API`: <span style="{style_mapping[IS_OPENAI_API_KEY_AVAILABLE]}">{availability_mapping[IS_OPENAI_API_KEY_AVAILABLE]}</span>.  `Cache`: <span style="{style_mapping[IS_CACHE_AVAILABLE]}">{availability_mapping[IS_CACHE_AVAILABLE]}</span>.''')
            file_output = gr.File(label="Output")

    clear_button_pp.click(fn=clear_inputs, inputs=[title, description_pp], outputs=[title, description_pp])
    submit_button_pp.click(fn=wrapped_generator, inputs=[title, description_pp, key], outputs=file_output)

demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch()
