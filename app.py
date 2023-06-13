import gradio as gr
import os
import openai
from auto_backgrounds import generate_backgrounds, generate_draft
from utils.file_operations import hash_name, list_folders
from references_generator import generate_top_k_references

# todo:
#   6. get logs when the procedure is not completed. *
#   7. 自己的文件库； 更多的prompts
#   2. 实现别的功能
#   3. Check API Key GPT-4 Support.
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
GPT4_ENBALE = os.getenv("GPT4_ENBALE")
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
    # except Exception as e:
    except openai.error.AuthenticationError:
        IS_OPENAI_API_KEY_AVAILABLE = False

DEFAULT_MODEL = "gpt-4" if IS_OPENAI_API_KEY_AVAILABLE else "gpt-3.5-turbo"
GPT4_INTERACTIVE = True if GPT4_ENBALE else False
DEFAULT_SECTIONS = ["introduction", "related works", "backgrounds", "methodology", "experiments",
                    "conclusion", "abstract"] if IS_OPENAI_API_KEY_AVAILABLE \
    else ["introduction", "related works"]

#######################################################################################################################
# Load the list of templates & knowledge databases
#######################################################################################################################
ALL_TEMPLATES = list_folders("latex_templates")
ALL_DATABASES = ["(None)"] + list_folders("knowledge_databases")

#######################################################################################################################
# Gradio UI
#######################################################################################################################
theme = gr.themes.Default(font=gr.themes.GoogleFont("Questrial"))
# .set(
#     background_fill_primary='#E5E4E2',
#     background_fill_secondary = '#F6F6F6',
#     button_primary_background_fill="#281A39"
# )
ANNOUNCEMENT = """
# Auto-Draft: 学术写作辅助工具

本Demo提供对[Auto-Draft](https://github.com/CCCBora/auto-draft)的auto_draft功能的测试. 
通过输入想要生成的论文名称（比如Playing atari with deep reinforcement learning)，即可由AI辅助生成论文模板.    

***2023-06-13 Update***:  
1. 新增‘高级选项-Prompts模式’. 这个模式仅会输出用于生成论文的Prompts而不会生成论文本身. 可以根据自己的需求修改Prompts, 也可以把Prompts复制给其他语言模型. 
2. 把默认的ICLR 2022模板改成了Default模板. 不再显示ICLR的页眉页尾.  
3. 使用GPT-4模型：
    - 点击Duplicate this Space, 进入Settings-> Repository secrets, 点击New Secret添加OPENAI_API_KEY为自己的OpenAI API Key. 添加GPT4_ENBALE为1. 
    - 或者可以访问[Auto-Draft-Private](https://huggingface.co/spaces/auto-academic/auto-draft-private).
    
如果有更多想法和建议欢迎加入QQ群里交流, 如果我在Space里更新了Key我会第一时间通知大家. 群号: ***249738228***."""

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
这一栏用于定义AI如何选取参考文献. 目前是两种方式混合:
1. GPT自动根据标题生成关键字，使用Semantic Scholar搜索引擎搜索文献，利用Specter获取Paper Embedding来自动选取最相关的文献作为GPT的参考资料.
2. 用户上传bibtex文件，使用Google Scholar搜索摘要作为GPT的参考资料. 
关于有希望利用本地文件来供GPT参考的功能将在未来实装.
"""

DOMAIN_KNOWLEDGE_INSTRUCTION = """### Domain Knowledge
这一栏用于定义AI的知识库. 将提供两种选择: 
1. 各个领域内由专家预先收集资料并构建的的FAISS向量数据库. 目前实装的数据库
* (None): 不使用任何知识库
* ml_textbook_test: 包含两本机器学习教材The Elements of Statistical Learning和Reinforcement Learning Theory and Algorithms. 仅用于测试知识库Pipeline.
2. 自行构建的使用OpenAI text-embedding-ada-002模型创建的FAISS向量数据库. (暂未实装)
"""

OUTPUTS_INSTRUCTION = """### Outputs
这一栏用于定义输出的内容：
* Template: 用于填装内容的LaTeX模板.
* Models: 使用GPT-4或者GPT-3.5-Turbo生成内容.
* Prompts模式: 不生成内容, 而是生成用于生成内容的Prompts. 可以手动复制到网页版或者其他语言模型中进行使用.
"""

OTHERS_INSTRUCTION = """### Others

"""

style_mapping = {True: "color:white;background-color:green",
                 False: "color:white;background-color:red"}  # todo: to match website's style
availability_mapping = {True: "AVAILABLE", False: "NOT AVAILABLE"}
STATUS = f'''## Huggingface Space Status  
 当`OpenAI API`显示AVAILABLE的时候这个Space可以直接使用.    
 当`OpenAI API`显示NOT AVAILABLE的时候这个Space可以通过在左侧输入OPENAI KEY来使用. 需要有GPT-4的API权限. 
 当`Cache`显示AVAILABLE的时候, 所有的输入和输出会被备份到我的云储存中. 显示NOT AVAILABLE的时候不影响实际使用. 
`OpenAI API`: <span style="{style_mapping[IS_OPENAI_API_KEY_AVAILABLE]}">{availability_mapping[IS_OPENAI_API_KEY_AVAILABLE]}</span>.  `Cache`: <span style="{style_mapping[IS_CACHE_AVAILABLE]}">{availability_mapping[IS_CACHE_AVAILABLE]}</span>.'''


def clear_inputs(*args):
    return "", ""


def clear_inputs_refs(*args):
    return "", 5


def wrapped_generator(
        paper_title, paper_description,  # main input
        openai_api_key=None, openai_url=None,  # key
        tldr=True, max_kw_refs=10, bib_refs=None, max_tokens_ref=2048,  # references
        knowledge_database=None, max_tokens_kd=2048, query_counts=10,  # domain knowledge
        paper_template="ICLR2022", selected_sections=None, model="gpt-4", prompts_mode=False,  # outputs parameters
        cache_mode=IS_CACHE_AVAILABLE  # handle cache mode
):
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
        from utils.storage import list_all_files, download_file
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
    try:
        output = generate_draft(
            paper_title, description=paper_description, # main input
           tldr=tldr, max_kw_refs=max_kw_refs, bib_refs=bib_refs, max_tokens_ref=max_tokens_ref,  # references
           knowledge_database=knowledge_database, max_tokens_kd=max_tokens_kd, query_counts=query_counts, # domain knowledge
           sections=selected_sections, model=model, template=paper_template, prompts_mode=prompts_mode, # outputs parameters
           )
        if cache_mode:
            from utils.storage import upload_file
            upload_file(output)
    except Exception as e:
        raise gr.Error(f"生成失败. Error: {e}")
    return output


def wrapped_references_generator(paper_title, num_refs, openai_api_key=None):
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        openai.Model.list()
    return generate_top_k_references(paper_title, top_k=num_refs)


with gr.Blocks(theme=theme) as demo:
    gr.Markdown(ANNOUNCEMENT)

    with gr.Row():
        with gr.Column(scale=2):
            key = gr.Textbox(value=openai_key, lines=1, max_lines=1, label="OpenAI Key",
                             visible=not IS_OPENAI_API_KEY_AVAILABLE)
            url = gr.Textbox(value=None, lines=1, max_lines=1, label="URL",
                             visible=False)
            # 每个功能做一个tab
            with gr.Tab("学术论文"):
                gr.Markdown(ACADEMIC_PAPER)

                title = gr.Textbox(value="Playing Atari with Deep Reinforcement Learning", lines=1, max_lines=1,
                                   label="Title", info="论文标题")

                description_pp = gr.Textbox(lines=5, label="Description (Optional)", visible=True,
                                            info="这篇论文的主要贡献和创新点. (生成所有章节时共享这个信息, 保持生成的一致性.)")

                with gr.Accordion("高级设置", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(OUTPUTS_INSTRUCTION)
                        with gr.Column(scale=2):
                            with gr.Row():
                                template = gr.Dropdown(label="Template", choices=ALL_TEMPLATES, value="Default",
                                                       interactive=True,
                                                       info="生成论文的模板.")
                                model_selection = gr.Dropdown(label="Model", choices=["gpt-4", "gpt-3.5-turbo"],
                                                              value=DEFAULT_MODEL,
                                                              interactive=GPT4_INTERACTIVE,
                                                              info="生成论文用到的语言模型.")
                                prompts_mode = gr.Checkbox(value=False, visible=True, interactive=True,
                                                             label="Prompts模式",
                                                             info="只输出用于生成论文的Prompts, 可以复制到别的地方生成论文.")

                            sections = gr.CheckboxGroup(
                                choices=["introduction", "related works", "backgrounds", "methodology", "experiments",
                                         "conclusion", "abstract"],
                                type="value", label="生成章节", interactive=True, info="选择生成论文的哪些章节.",
                                value=DEFAULT_SECTIONS)

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(REFERENCES_INSTRUCTION)

                        with gr.Column(scale=2):
                            max_kw_ref_slider = gr.Slider(minimum=1, maximum=20, value=10, step=1,
                                                       interactive=True, label="MAX_KW_REFS",
                                                       info="每个Keyword搜索几篇参考文献", visible=False)

                            max_tokens_ref_slider = gr.Slider(minimum=256, maximum=4096, value=2048, step=2,
                                                       interactive=True, label="MAX_TOKENS",
                                                       info="参考文献内容占用Prompts中的Token数")

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
                            query_counts_slider = gr.Slider(minimum=1, maximum=20, value=10, step=1,
                                                       interactive=True, label="QUERY_COUNTS",
                                                       info="从知识库内检索多少条内容", visible=False)
                            max_tokens_kd_slider = gr.Slider(minimum=256, maximum=4096, value=2048, step=2,
                                                      interactive=True, label="MAX_TOKENS",
                                                      info="知识库内容占用Prompts中的Token数")
                            # template = gr.Dropdown(label="Template", choices=ALL_TEMPLATES, value="Default",
                            #                        interactive=True,
                            #                        info="生成论文的参考模板.")
                            domain_knowledge = gr.Dropdown(label="预载知识库",
                                                           choices=ALL_DATABASES,
                                                           value="(None)",
                                                           interactive=True,
                                                           info="使用预先构建的知识库.")
                            local_domain_knowledge = gr.File(label="本地知识库 (暂未实装)", interactive=False)
                with gr.Row():
                    clear_button_pp = gr.Button("Clear")
                    submit_button_pp = gr.Button("Submit", variant="primary")

            # with gr.Tab("文献搜索"):
            #     gr.Markdown(REFERENCES)
            #
            #     title_refs = gr.Textbox(value="Playing Atari with Deep Reinforcement Learning", lines=1, max_lines=1,
            #                             label="Title", info="论文标题")
            #     slider_refs = gr.Slider(minimum=1, maximum=100, value=5, step=1,
            #                             interactive=True, label="最相关的参考文献数目")
            #     with gr.Row():
            #         clear_button_refs = gr.Button("Clear")
            #         submit_button_refs = gr.Button("Submit", variant="primary")

            with gr.Tab("文献综述 (Coming soon!)"):
                gr.Markdown('''
                <h1  style="text-align: center;">Coming soon!</h1>
                ''')
            with gr.Tab("Github文档 (Coming soon!)"):
                gr.Markdown('''
                <h1  style="text-align: center;">Coming soon!</h1>
                ''')

        with gr.Column(scale=1):
            gr.Markdown(STATUS)
            file_output = gr.File(label="Output")
            json_output = gr.JSON(label="References")


    # def wrapped_generator(
    #         paper_title, paper_description,  # main input
    #         openai_api_key=None, openai_url=None,  # key
    #         tldr=True, max_kw_refs=10, bib_refs=None, max_tokens_ref=2048,  # references
    #         knowledge_database=None, max_tokens_kd=2048, query_counts=10,  # domain knowledge
    #         paper_template="ICLR2022", selected_sections=None, model="gpt-4", prompts_mode=False,  # outputs parameters
    #         cache_mode=IS_CACHE_AVAILABLE  # handle cache mode
    # ):
    clear_button_pp.click(fn=clear_inputs, inputs=[title, description_pp], outputs=[title, description_pp])
    submit_button_pp.click(fn=wrapped_generator,
                           inputs=[title, description_pp, key, url,
                                   tldr_checkbox, max_kw_ref_slider,  bibtex_file, max_tokens_ref_slider,
                                   domain_knowledge, max_tokens_kd_slider, query_counts_slider,
                                   template, sections, model_selection, prompts_mode], outputs=file_output)

    # clear_button_refs.click(fn=clear_inputs_refs, inputs=[title_refs, slider_refs], outputs=[title_refs, slider_refs])
    # submit_button_refs.click(fn=wrapped_references_generator,
    #                          inputs=[title_refs, slider_refs, key], outputs=json_output)

demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch(show_error=True)
