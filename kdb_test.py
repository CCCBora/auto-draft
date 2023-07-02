from utils.knowledge import Knowledge
from langchain.vectorstores import FAISS
from utils.file_operations import list_folders
from huggingface_hub import snapshot_download
import gradio as gr
import os
import json
from models import EMBEDDINGS

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("KDB_REPO")

snapshot_download(REPO_ID, repo_type="dataset", local_dir="knowledge_databases/",
                  local_dir_use_symlinks=False, token=HF_TOKEN)
ALL_KDB = ["(None)"] + list_folders("knowledge_databases")



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

ANNOUNCEMENT = """"""

with gr.Blocks() as demo:
    gr.HTML(ANNOUNCEMENT)
    with gr.Row():
        with gr.Column():
            kdb_dropdown = gr.Dropdown(choices=ALL_KDB, value="(None)")
            user_input = gr.Textbox(label="Input")
            button_retrieval = gr.Button("Query", variant="primary")

            with gr.Accordion("Advanced Setting", open=False):
                query_counts_slider = gr.Slider(minimum=1, maximum=20, value=10, step=1,
                                                       interactive=True, label="QUERY_COUNTS",
                                                       info="从知识库内检索多少条内容")

        retrieval_output = gr.JSON(label="Output")


    button_retrieval.click(fn=query_from_kdb, inputs=[user_input, kdb_dropdown, query_counts_slider], outputs=[retrieval_output, user_input])
demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch(show_error=True)


