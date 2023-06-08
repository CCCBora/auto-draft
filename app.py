import gradio as gr
import os
import openai
from auto_backgrounds import generate_backgrounds, generate_draft
from utils.file_operations import hash_name, list_folders
from references_generator import generate_top_k_references

# todo:
#   6. get logs when the procedure is not completed. *
#   7. 自己的文件库； 更多的prompts
#   8. Decide on how to generate the main part of a paper * (Langchain/AutoGPT
#   1. 把paper改成纯JSON?
#   2. 实现别的功能
#   3. Check API Key GPT-4 Support.
#   8. Re-build some components using `langchain`
#           - in `gpt_interation`, use LLM
# future:
#   generation.log sometimes disappears (ignore this)
#   1. Check if there are any duplicated citations
#   2. Remove potential thebibliography and bibitem in .tex file

#######################################################################################################################
# Check if openai and cloud storage available
#######################################################################################################################
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

ALL_TEMPLATES = list_folders("latex_templates")


def clear_inputs(*args):
    return "", ""

def clear_inputs_refs(*args):
    return "", 5


def wrapped_generator(paper_title, paper_description, openai_api_key=None,
                      paper_template="ICLR2022", tldr=True, selected_sections=None, bib_refs=None, model="gpt-4",
                      cache_mode=IS_CACHE_AVAILABLE):
    # if `cache_mode` is True, then follow the following steps:
    #        check if "title"+"description" have been generated before
    #        if so, download from the cloud storage, return it
    #        if not, generate the result.
    if bib_refs is not None:
        bib_refs = bib_refs.name
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        try:
            openai.Model.list()
        except Exception as e:
            raise gr.Error(f"Key错误. Error: {e}")

    if cache_mode:
        from utils.storage import list_all_files, download_file, upload_file
        # check if "title"+"description" have been generated before
        input_dict = {"title": paper_title, "description": paper_description,
                      "generator": "generate_draft"}
        file_name = hash_name(input_dict) + ".zip"
        file_list = list_all_files()
        # print(f"{file_name} will be generated. Check the file list {file_list}")
        if file_name in file_list:
            # download from the cloud storage, return it
            download_file(file_name)
            return file_name
        else:
            try:
                # generate the result.
                # output = fake_generate_backgrounds(title, description, openai_key)
                output = generate_draft(paper_title, paper_description, template=paper_template,
                                        tldr=tldr, sections=selected_sections, bib_refs=bib_refs, model=model)
                # output = generate_draft(paper_title, paper_description, template, "gpt-4")
                upload_file(output)
                return output
            except Exception as e:
                raise gr.Error(f"生成失败. Error {e.__name__}: {e}")
    else:
        try:
            # output = fake_generate_backgrounds(title, description, openai_key)
            output = generate_draft(paper_title, paper_description, template=paper_template,
                                    tldr=tldr, sections=selected_sections, bib_refs=bib_refs, model=model)
        except Exception as e:
            raise gr.Error(f"生成失败. Error: {e}")
        return output


def wrapped_references_generator(paper_title, num_refs, openai_api_key=None):
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        openai.Model.list()
    return generate_top_k_references(paper_title, top_k=num_refs)



theme = gr.themes.Default(font=gr.themes.GoogleFont("Questrial"))
# .set(
#     background_fill_primary='#E5E4E2',
#     background_fill_secondary = '#F6F6F6',
#     button_primary_background_fill="#281A39"
# )

ACADEMIC_PAPER = """## 一键生成论文初稿

1. 在Title文本框中输入想要生成的论文名称（比如Playing Atari with Deep Reinforcement Learning). 
2. 点击Submit. 等待大概十五分钟(全文). 
3. 在右侧下载.zip格式的输出，在Overleaf上编译浏览.  
"""


REFERENCES = """## 一键搜索相关论文
(此功能已经被整合进一键生成论文初稿)
1. 在Title文本框中输入想要搜索文献的论文（比如Playing Atari with Deep Reinforcement Learning). 
2. 点击Submit. 等待大概十分钟. 
3. 在右侧JSON处会显示相关文献.  
"""

REFERENCES_INSTRUCTION = """### References
这一行用于定义AI如何选取参考文献. 目前是两种方式混合:
1. GPT自动根据标题生成关键字，使用Semantic Scholar搜索引擎搜索文献，利用Specter获取Paper Embedding来自动选取最相关的文献作为GPT的参考资料.
2. 用户上传bibtex文件，使用Google Scholar搜索摘要作为GPT的参考资料. 
关于有希望利用本地文件来供GPT参考的功能将在未来实装.
"""

DOMAIN_KNOWLEDGE_INSTRUCTION = """### Domain Knowledge
(暂未实装)
这一行用于定义AI的知识库. 将提供两种选择: 
1. 各个领域内由专家预先收集资料并构建的的FAISS向量数据库. 每个数据库内包含了数百万页经过同行评议的论文和专业经典书籍. 
2. 自行构建的使用OpenAI text-embedding-ada-002模型创建的FAISS向量数据库.
"""

OTHERS_INSTRUCTION = """### Others

"""


with gr.Blocks(theme=theme) as demo:
    gr.Markdown('''
    # Auto-Draft: 文献整理辅助工具
    
    本Demo提供对[Auto-Draft](https://github.com/CCCBora/auto-draft)的auto_draft功能的测试. 
    通过输入想要生成的论文名称（比如Playing atari with deep reinforcement learning)，即可由AI辅助生成论文模板.    
    
    ***2023-06-08 Update***: 
    * 目前对英文的生成效果更好. 如果需要中文文章可以使用[GPT学术优化](https://github.com/binary-husky/gpt_academic)的`Latex全文翻译、润色`功能. 
    * GPT3.5模型可能会因为Token数不够导致一部分章节为空. 可以在高级设置里减少生成的章节. 
    
    ***2023-05-17 Update***: 我的API的余额用完了, 所以这个月不再能提供GPT-4的API Key. 这里为大家提供了一个位置输入OpenAI API Key. 同时也提供了GPT-3.5的兼容. 欢迎大家自行体验. 
    
    如果有更多想法和建议欢迎加入QQ群里交流, 如果我在Space里更新了Key我会第一时间通知大家. 群号: ***249738228***.
    ''')

    with gr.Row():
        with gr.Column(scale=2):
            key = gr.Textbox(value=openai_key, lines=1, max_lines=1, label="OpenAI Key",
                             visible=not IS_OPENAI_API_KEY_AVAILABLE)

            # generator = gr.Dropdown(choices=["学术论文", "文献总结"], value="文献总结",
            # label="Selection", info="目前支持生成'学术论文'和'文献总结'.", interactive=True)

            # 每个功能做一个tab
            with gr.Tab("学术论文"):
                gr.Markdown(ACADEMIC_PAPER)

                title = gr.Textbox(value="Playing Atari with Deep Reinforcement Learning", lines=1, max_lines=1,
                                   label="Title", info="论文标题")
                with gr.Accordion("高级设置", open=False):
                    with gr.Row():
                        description_pp = gr.Textbox(lines=5, label="Description (Optional)", visible=True,
                                                    info="对希望生成的论文的一些描述. 包括这篇论文的创新点, 主要贡献, 等.")
                        with gr.Row():
                            template = gr.Dropdown(label="Template", choices=ALL_TEMPLATES, value="Default",
                                                   interactive=True,
                                                   info="生成论文的参考模板.")
                            model_selection = gr.Dropdown(label="Model", choices=["gpt-4", "gpt-3.5-turbo"],
                                                          value="gpt-3.5-turbo",
                                                          interactive=True,
                                                          info="生成论文用到的语言模型.")
                        sections = gr.CheckboxGroup(
                            choices=["introduction", "related works", "backgrounds", "methodology", "experiments",
                                     "conclusion", "abstract"],
                            type="value", label="生成章节", interactive=True,
                            value=["introduction", "related works"])

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(REFERENCES_INSTRUCTION)

                        with gr.Column(scale=2):
                            search_engine = gr.Dropdown(label="Search Engine",
                                                        choices=["ArXiv", "Semantic Scholar", "Google Scholar", "None"],
                                                        value="Semantic Scholar",
                                                        interactive=False,
                                                        visible=False,
                                                        info="用于决定GPT用什么搜索引擎来搜索文献. (暂不支持修改)")
                            tldr_checkbox = gr.Checkbox(value=True, label="TLDR;",
                                                        info="选择此筐表示将使用Semantic Scholar的TLDR作为文献的总结.",
                                                        interactive=True)
                            gr.Markdown('''
                            上传.bib文件提供AI需要参考的文献. 
                            ''')
                            bibtex_file = gr.File(label="Upload .bib file", file_types=["text"],
                                                  interactive=True)

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(DOMAIN_KNOWLEDGE_INSTRUCTION)

                        with gr.Column(scale=2):
                            domain_knowledge = gr.Dropdown(label="预载知识库",
                                                        choices=["(None)", "Machine Learning"],
                                                        value="(None)",
                                                        interactive=False,
                                                        info="使用预先构建的知识库. (暂未实装)")
                            local_domain_knowledge = gr.File(label="本地知识库 (暂未实装)", interactive=False)
                with gr.Row():
                    clear_button_pp = gr.Button("Clear")
                    submit_button_pp = gr.Button("Submit", variant="primary")

            with gr.Tab("文献搜索"):
                gr.Markdown(REFERENCES)

                title_refs = gr.Textbox(value="Playing Atari with Deep Reinforcement Learning", lines=1, max_lines=1,
                                   label="Title", info="论文标题")
                slider_refs = gr.Slider(minimum=1, maximum=100, value=5, step=1,
                                   interactive=True, label="最相关的参考文献数目")
                with gr.Row():
                    clear_button_refs = gr.Button("Clear")
                    submit_button_refs = gr.Button("Submit", variant="primary")

            with gr.Tab("文献综述 (Coming soon!)"):
                gr.Markdown('''
                <h1  style="text-align: center;">Coming soon!</h1>
                ''')
            with gr.Tab("Github文档 (Coming soon!)"):
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
            json_output = gr.JSON(label="References")

    clear_button_pp.click(fn=clear_inputs, inputs=[title, description_pp], outputs=[title, description_pp])
    submit_button_pp.click(fn=wrapped_generator,
                           inputs=[title, description_pp, key, template, tldr_checkbox, sections, bibtex_file,
                                   model_selection], outputs=file_output)

    clear_button_refs.click(fn=clear_inputs_refs, inputs=[title_refs, slider_refs], outputs=[title_refs, slider_refs])
    submit_button_refs.click(fn=wrapped_references_generator,
                           inputs=[title_refs, slider_refs, key], outputs=json_output)

demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch(show_error=True)
