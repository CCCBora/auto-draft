import uuid
import gradio as gr
import os
import openai
import yaml
from utils.file_operations import list_folders, urlify
from huggingface_hub import snapshot_download
from wrapper import generator_wrapper

# future:
#   generation.log sometimes disappears (ignore this)
#   1. Check if there are any duplicated citations
#   2. Remove potential thebibliography and bibitem in .tex file

#######################################################################################################################
# Environment Variables
#######################################################################################################################
# OPENAI_API_KEY: OpenAI API key for GPT models
# OPENAI_API_BASE: (Optional) Support alternative OpenAI minors
# GPT4_ENABLE: (Optional) Set it to 1 to enable GPT-4 model.

# AWS_ACCESS_KEY_ID: (Optional)
#   Access AWS cloud storage (you need to edit `BUCKET_NAME` in `utils/storage.py` if you need to use this function)
# AWS_SECRET_ACCESS_KEY: (Optional)
#   Access AWS cloud storage (you need to edit `BUCKET_NAME` in `utils/storage.py` if you need to use this function)
# KDB_REPO: (Optional) A Huggingface dataset hosting Knowledge Databases
# HF_TOKEN: (Optional) Access to KDB_REPO

#######################################################################################################################
# Check if openai and cloud storage available
#######################################################################################################################
openai_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
if openai_api_base is not None:
    openai.api_base = openai_api_base
GPT4_ENABLE = os.getenv("GPT4_ENABLE")  # disable GPT-4 for public repo

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
    # except Exception as e:
    except openai.error.AuthenticationError:
        IS_OPENAI_API_KEY_AVAILABLE = False

DEFAULT_MODEL = "gpt-4" if GPT4_ENABLE else 'gpt-3.5-turbo-16k'
GPT4_INTERACTIVE = True if GPT4_ENABLE else False
DEFAULT_SECTIONS = ["introduction", "related works", "backgrounds", "methodology", "experiments",
                    "conclusion", "abstract"] if GPT4_ENABLE \
    else ["introduction", "related works"]

MODEL_LIST = ['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k']

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("KDB_REPO")
if HF_TOKEN is not None and REPO_ID is not None:
    snapshot_download(REPO_ID, repo_type="dataset", local_dir="knowledge_databases/",
                      local_dir_use_symlinks=False, token=HF_TOKEN)
    KDB_LIST = ["(None)"] + list_folders("knowledge_databases")

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

本Demo提供对[Auto-Draft](https://github.com/CCCBora/auto-draft)的学术论文模板生成功能的测试. 学术综述和Github文档功能正在开发中. 

## 主要功能
通过输入想要生成的论文名称（比如Playing atari with deep reinforcement learning)，即可由AI辅助生成论文模板.     

***2023-06-13 Update***: 
- 增加了最新的gpt-3.5-turbo-16k模型的支持.  

***2023-06-13 Update***:  
1. 新增‘高级选项-Prompts模式’. 这个模式仅会输出用于生成论文的Prompts而不会生成论文本身. 可以根据自己的需求修改Prompts, 也可以把Prompts复制给其他语言模型. 
2. 把默认的ICLR 2022模板改成了Default模板. 不再显示ICLR的页眉页尾.  
3. 中文支持: 暂不支持. 建议使用英文生成论文, 然后把输出结果送入[GPT 学术优化](https://github.com/binary-husky/gpt_academic)中的Latex全文翻译、润色功能即可. 
4. 使用GPT-4模型：
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
2. 用户通过输入文章标题(用英文逗号隔开), AI会自动搜索文献作为参考资料.
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
* Prompts模式: 不生成内容, 而是生成用于生成内容的Prompts. 可以手动复制到网页版或者其他语言模型中进行使用. (放在输出的ZIP文件的prompts.json文件中)
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
        openai_api_key=None,  # key
        tldr=True, max_kw_refs=10, refs=None, max_tokens_ref=2048,  # references
        knowledge_database=None, max_tokens_kd=2048, query_counts=10,  # domain knowledge
        paper_template="ICLR2022", selected_sections=None, model="gpt-4", prompts_mode=False,  # outputs parameters
        cache_mode=IS_CACHE_AVAILABLE  # handle cache mode
):
    file_name_upload = urlify(paper_title) + "_" + uuid.uuid1().hex + ".zip"

    # load the default configuration file
    with open("configurations/default.yaml", 'r') as file:
        config = yaml.safe_load(file)
    config["paper"]["title"] = paper_title
    config["paper"]["description"] = paper_description
    config["references"]["tldr"] = tldr
    config["references"]["max_kw_refs"] = max_kw_refs
    config["references"]["refs"] = refs
    config["references"]["max_tokens_ref"] = max_tokens_ref
    config["domain_knowledge"]["knowledge_database"] = knowledge_database
    config["domain_knowledge"]["max_tokens_kd"] = max_tokens_kd
    config["domain_knowledge"]["query_counts"] = query_counts
    config["output"]["selected_sections"] = selected_sections
    config["output"]["model"] = model
    config["output"]["template"] = paper_template
    config["output"]["prompts_mode"] = prompts_mode

    if openai_api_key is not None:
        openai.api_key = openai_api_key
        try:
            openai.Model.list()
        except Exception as e:
            raise gr.Error(f"Key错误. Error: {e}")
    try:
        output = generator_wrapper(config)
        if cache_mode:
            from utils.storage import upload_file
            upload_file(output, target_name=file_name_upload)
    except Exception as e:
        raise gr.Error(f"生成失败. Error: {e}")
    return output


with gr.Blocks(theme=theme) as demo:
    gr.Markdown(ANNOUNCEMENT)

    with gr.Row():
        with gr.Column(scale=2):
            key = gr.Textbox(value=openai_key, lines=1, max_lines=1, label="OpenAI Key",
                             visible=not IS_OPENAI_API_KEY_AVAILABLE)
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
                                model_selection = gr.Dropdown(label="Model", choices=MODEL_LIST,
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

                            max_tokens_ref_slider = gr.Slider(minimum=256, maximum=8192, value=2048, step=2,
                                                              interactive=True, label="MAX_TOKENS",
                                                              info="参考文献内容占用Prompts中的Token数")

                            tldr_checkbox = gr.Checkbox(value=True, label="TLDR;",
                                                        info="选择此筐表示将使用Semantic Scholar的TLDR作为文献的总结.",
                                                        interactive=True)

                            text_ref = gr.Textbox(lines=5, label="References (Optional)", visible=True,
                                                  info="交给AI参考的文献的标题, 用英文逗号`,`隔开.")

                            gr.Examples(
                                examples = ["Understanding the Impact of Model Incoherence on Convergence of Incremental SGD with Random Reshuffle,"
                                            "Variance-Reduced Off-Policy TDC Learning: Non-Asymptotic Convergence Analysis,"
                                            "Greedy-GQ with Variance Reduction: Finite-time Analysis and Improved Complexity"],
                                inputs=text_ref,
                                cache_examples=False
                            )

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown(DOMAIN_KNOWLEDGE_INSTRUCTION)

                        with gr.Column(scale=2):
                            query_counts_slider = gr.Slider(minimum=1, maximum=20, value=10, step=1,
                                                            interactive=True, label="QUERY_COUNTS",
                                                            info="从知识库内检索多少条内容", visible=False)
                            max_tokens_kd_slider = gr.Slider(minimum=256, maximum=8192, value=2048, step=2,
                                                             interactive=True, label="MAX_TOKENS",
                                                             info="知识库内容占用Prompts中的Token数")
                            domain_knowledge = gr.Dropdown(label="预载知识库",
                                                           choices=ALL_DATABASES,
                                                           value="(None)",
                                                           interactive=True,
                                                           info="使用预先构建的知识库.")
                            local_domain_knowledge = gr.File(label="本地知识库 (暂未实装)", interactive=False)
                with gr.Row():
                    clear_button_pp = gr.Button("Clear")
                    submit_button_pp = gr.Button("Submit", variant="primary")
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
    clear_button_pp.click(fn=clear_inputs, inputs=[title, description_pp], outputs=[title, description_pp])
    submit_button_pp.click(fn=wrapped_generator,
                           inputs=[title, description_pp, key,
                                   tldr_checkbox, max_kw_ref_slider, text_ref, max_tokens_ref_slider,
                                   domain_knowledge, max_tokens_kd_slider, query_counts_slider,
                                   template, sections, model_selection, prompts_mode], outputs=file_output)

demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch(show_error=True)
