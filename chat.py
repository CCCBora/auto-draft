import gradio as gr
import os
import openai
from utils.references import References
from utils.gpt_interaction import GPTModel
from utils.prompts import SYSTEM

openai_key = os.getenv("OPENAI_API_KEY")
default_model = os.getenv("DEFAULT_MODEL")
if default_model is None:
    default_model = "gpt-3.5-turbo-16k"

openai.api_key = openai_key

paper_system_prompt =  '''You are an assistant designed to propose choices of research direction.
The user will input questions or some keywords of a fields. You need to generate some paper titles and main contributions. Ensure follow the following instructions:
Instruction:
- Your response should follow the JSON format.
- Your response should have the following structure: 
{
    "your suggested paper title": 
        {
            "summary": "an overview introducing what this paper will include",
            "contributions": {
                "contribution1": {"statement": "briefly describe this contribution", "reason": "reason why this contribution can make this paper outstanding"}, 
                "contribution2": {"statement": "briefly describe this contribution", "reason": "reason why this contribution can make this paper outstanding"}, 
                ...
            }
        }
    "your suggested paper title": 
        {
            "summary": "an overview introducing what this paper will include",
            "contributions": {
                "contribution1": {"statement": "briefly describe this contribution", "reason": "reason why this contribution can make this paper outstanding"}, 
                "contribution2": {"statement": "briefly describe this contribution", "reason": "reason why this contribution can make this paper outstanding"}, 
                ...
            }
        }
    ...
}
- Please list three to five suggested title and at least three contributions for each paper.
'''


contribution_system_prompt = '''You are an assistant designed to criticize the contributions of a paper. You will be provided Paper's Title, References and Contributions. Ensure follow the following instructions:
Instruction:
- Your response should follow the JSON format.
- Your response should have the following structure: 
{
    "title": "the title provided by the user",
    "comment": "your thoughts on if this title clearly reflects the key ideas of this paper and explain why"
    "contributions": {
        "contribution1": {"statement": "briefly describe what the contribution is", 
                          "reason": "reason why the user claims it is a contribution", 
                          "judge": "your thought about if this is a novel contribution and explain why", 
                          "suggestion": "your suggestion on how to modify the research direction to enhance the novelty "},
        "contribution2": {"statement": "briefly describe what the contribution is", 
                          "reason": "reason why the user claims it is a contribution", 
                          "judge": "your thought about if this is a novel contribution and explain why", 
                          "suggestion": "your suggestion on how to modify the research direction to enhance the novelty "},
        ...
    }
} 
- You need to carefully check if the claimed contribution has been made in the provided references, which makes the contribution not novel.  
- You also need to propose your concerns on if any of contributions could be incremental or just a mild modification on an existing work.
'''


ANNOUNCEMENT = """
# Paper Bot

Criticize your paper's contribution by searching related references online! This nice bot will also give you some suggestions.
"""

def criticize_my_idea(title, contributions, max_tokens=4096):
    ref = References(title=title, description=f"{contributions}")
    keywords, _ = llm(systems=SYSTEM["keywords"], prompts=title, return_json=True)
    keywords = {keyword: 10 for keyword in keywords}
    ref.collect_papers(keywords)
    ref_prompt = ref.to_prompts(max_tokens=max_tokens)

    prompt = f"Title: {title}\n References: {ref_prompt}\n Contributions: {contributions}"
    output, _ = llm(systems=contribution_system_prompt, prompts=prompt, return_json=True)
    return output, ref_prompt

def paste_title(suggestions):
    if suggestions:
        title = suggestions['title']['new title']
        contributions = suggestions['contributions']

        return title, contributions, {}, {}, {}
    else:
        return "", "", {}, {}, {}

def generate_choices(thoughts):
    output, _ = llm(systems=paper_system_prompt, prompts=thoughts, return_json=True)
    return output



with gr.Blocks() as demo:
    llm = GPTModel(model=default_model)
    gr.Markdown(ANNOUNCEMENT)

    with gr.Row():
        with gr.Tab("Make it an idea!"):
            thoughts_input = gr.Textbox(label="Thoughts")
            with gr.Accordion("Show prompts", open=False):
                prompts_1 = gr.Textbox(label="Prompts", interactive=False, value=paper_system_prompt)

            with gr.Row():
                button_generate_idea = gr.Button("Make it an idea!", variant="primary")

        with gr.Tab("Criticize my idea!"):
            title_input = gr.Textbox(label="Title")
            contribution_input = gr.Textbox(label="Contributions", lines=5)
            with gr.Accordion("Show prompts", open=False):
                prompts_2 = gr.Textbox(label="Prompts", interactive=False, value=contribution_system_prompt)

            with gr.Row():
                button_submit = gr.Button("Criticize my idea!", variant="primary")

        with gr.Tab("Make it a paper!"):
            gr.Markdown("## Coming Soon!")

        with gr.Column(scale=1):
            contribution_output = gr.JSON(label="Contributions")
            with gr.Accordion("References", open=False):
                references_output = gr.JSON(label="References")

    button_submit.click(fn=criticize_my_idea, inputs=[title_input, contribution_input], outputs=[contribution_output, references_output])
    button_generate_idea.click(fn=generate_choices, inputs=thoughts_input, outputs=contribution_output)
demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch(show_error=True)
