from utils.knowledge import Knowledge
from langchain.vectorstores import FAISS
from utils.file_operations import list_folders
from huggingface_hub import snapshot_download
import gradio as gr
import os
import json
from models import EMBEDDINGS
from utils.gpt_interaction import GPTModel
from utils.prompts import SYSTEM
import openai

llm = GPTModel(model="gpt-3.5-turbo")
openai.api_key = os.getenv("OPENAI_API_KEY")

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("KDB_REPO")
if HF_TOKEN is not None and REPO_ID is not None:
    snapshot_download(REPO_ID, repo_type="dataset", local_dir="knowledge_databases/",
                      local_dir_use_symlinks=False, token=HF_TOKEN)
ALL_KDB = ["(None)"] + list_folders("knowledge_databases")

ANNOUNCEMENT = """
# Evaluate the quality of retrieved date from the FAISS database

Use this space test the performance of some pre-constructed vector databases hosted at `shaocongma/kdb`. To use this space for your own FAISS database, follow this instruction:
1. Duplicate this space.
2. Add the secret key `HF_TOKEN` with your own Huggingface User Access Token. 
3. Create a Huggingface Dataset. Put your FAISS database to it.
4. Add the secret key `REPO_ID` as your dataset's address. 
"""
AUTODRAFT = """
AutoDraft is a GPT-based project to generate an academic paper using the title and contributions. When generating specific sections, AutoDraft will query some necessary backgrounds in related fields from the pre-constructed vector database.  
"""

def query_from_kdb(input, kdb, query_counts):
    if kdb == "(None)":
        return {"knowledge_database": "(None)", "input": input, "output": ""}, ""

    db_path = f"knowledge_databases/{kdb}"
    db_config_path = os.path.join(db_path, "db_meta.json")
    db_index_path = os.path.join(db_path, "faiss_index")
    if os.path.isdir(db_path):
        # load configuration file
        with open(db_config_path, "r", encoding="utf-8") as f:
            db_config = json.load(f)
        model_name = db_config["embedding_model"]
        embeddings = EMBEDDINGS[model_name]
        db = FAISS.load_local(db_index_path, embeddings)
        knowledge = Knowledge(db=db)
        knowledge.collect_knowledge({input: query_counts}, max_query=query_counts)
        domain_knowledge = knowledge.to_json()
    else:
        raise RuntimeError(f"Failed to query from FAISS.")
    return domain_knowledge, ""

def query_from_kdb_llm(title, contributions, kdb, query_counts):
    if kdb == "(None)":
        return {"knowledge_database": "(None)", "title": title, "contributions": contributions, "output": ""}, "", {}

    db_path = f"knowledge_databases/{kdb}"
    db_config_path = os.path.join(db_path, "db_meta.json")
    db_index_path = os.path.join(db_path, "faiss_index")
    if os.path.isdir(db_path):
        # load configuration file
        with open(db_config_path, "r", encoding="utf-8") as f:
            db_config = json.load(f)
        model_name = db_config["embedding_model"]
        embeddings = EMBEDDINGS[model_name]
        db = FAISS.load_local(db_index_path, embeddings)
        knowledge = Knowledge(db=db)
        prompts = f"Title: {title}\n Contributions: {contributions}"
        preliminaries_kw, _ = llm(systems=SYSTEM["preliminaries"], prompts=prompts, return_json=True)
        knowledge.collect_knowledge(preliminaries_kw, max_query=query_counts)
        domain_knowledge = knowledge.to_json()
    else:
        raise RuntimeError(f"Failed to query from FAISS.")
    return domain_knowledge, "", preliminaries_kw

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(ANNOUNCEMENT)

            kdb_dropdown = gr.Dropdown(choices=ALL_KDB, value="(None)", label="Knowledge Databases",
                                       info="Pre-defined knowledge databases utilized to aid in the generation of academic writing content. "
                                            "Hosted at `shaocongma/kdb`.")
            with gr.Tab("User's Input"):
                user_input = gr.Textbox(label="Input", info="Input anything you like to test what will be retrived from the vector database.")
                with gr.Row():
                    button_clear = gr.Button("Clear")
                    button_retrieval = gr.Button("Retrieve", variant="primary")
            with gr.Tab("AutoDraft"):
                gr.Markdown(AUTODRAFT)
                title_input = gr.Textbox(label="Title")
                contribution_input = gr.Textbox(label="Contributions", lines=5)
                with gr.Row():
                    button_clear_2 = gr.Button("Clear")
                    button_retrieval_2 = gr.Button("Retrieve", variant="primary")

            with gr.Accordion("Advanced Setting", open=False):
                query_counts_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1,
                                                interactive=True, label="QUERY_COUNTS",
                                                info="How many contents will be retrieved from the vector database.")

        with gr.Column():
            retrieval_output = gr.JSON(label="Output")
            llm_kws = gr.JSON(label="Keywords generated by LLM")

    button_retrieval.click(fn=query_from_kdb,
                           inputs=[user_input, kdb_dropdown, query_counts_slider],
                           outputs=[retrieval_output, user_input])
    button_retrieval_2.click(fn=query_from_kdb_llm,
                             inputs=[title_input, contribution_input, kdb_dropdown, query_counts_slider],
                             outputs=[retrieval_output, user_input, llm_kws])

demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch(show_error=True)


