import gradio as gr
import os
import openai
from utils.references import References
from utils.gpt_interaction import GPTModel
from utils.prompts import SYSTEM

openai_key = os.getenv("OPENAI_API_KEY")
default_model = os.getenv("DEFAULT_MODEL")
openai.api_key = openai_key

contribution_system_prompt_1 = '''You are an assistant designed to propose potential contributions of a given title of the paper. Ensure follow the following instructions:
Instruction:
- Your response should follow the JSON format.
- Your response should have the following structure: {"contribution1": {"statement": "briefly describe what the contribution is", "reason": "reason why this contribution has not been made by other literatures"}, "contribution2": {"statement": "briefly describe what the contribution is", "reason": "reason why this contribution has not been made by other literatures"}, ...}'''

contribution_system_prompt_2 = '''You are an assistant designed to criticize the contributions of a paper. You will be provided Paper's Title, References and Contributions. Ensure follow the following instructions:
Instruction:
- Your response should follow the JSON format.
- Your response should have the following structure: 
{"contribution1": {"statement": "briefly describe what the contribution is", "reason": "reason why this becomes a contribution from the user", "think":"your thought about if this is a novel contribution", "criticism": "reason why this can or cannot be a novel contribution"}, 
"contribution2": {"statement": "briefly describe what the contribution is", "reason": "reason why this becomes a contribution from the user", "think":"your thought about if this is a novel contribution", "criticism": "reason why this can or cannot be a novel contribution"}, ...}
- You need to carefully check if the claimed contribution has been made in the provided references, which makes the contribution not novel.  
'''

suggestions_system_prompt = '''You are an assistant designed to help improve the novelty of a paper. You will be provided Paper's Title, References, Criticism, and Contributions. Ensure follow the following instructions:
Instruction:
- Your response should follow the JSON format.
- Your response should have the following structure: 
{"title": {"suggestion": "your suggestion on the title", "new title": "your suggested title based on your suggestion", "reason": "your reason why you want to make such modification based on the references and criticism"},
"contributions": {"new contribution 1": {"statement": "your proposed new contribution", "reason": "why this is a novel contribution"}, "new contribution 2": {"statement": "your proposed new contribution", "reason": "why this is a novel contribution"}, ...}}
- Your reason should be based on the references and solve the criticism.
'''


ANNOUNCEMENT = """
# Paper Bot

Criticize your paper's contribution by searching related references online! This nice bot will also give you some suggestions.
"""


def reset():
    return "", "", {}, {}, {}


def search_refs(title, contributions):
    ref = References(title=title, description=contributions)

    keywords, _ = llm(systems=SYSTEM["keywords"], prompts=title, return_json=True)
    keywords = {keyword: 10 for keyword in keywords}

    ref.collect_papers(keywords)
    return ref.to_prompts(max_tokens=8192)


def criticize_my_idea(title, contributions, refined_contributions, suggestions):
    if refined_contributions:
        cont = {k: {"statement": v["statement"]} for k, v in refined_contributions.items()}

        ref = References(title=title, description=f"{cont}")
        keywords, _ = llm(systems=SYSTEM["keywords"], prompts=title, return_json=True)
        keywords = {keyword: 10 for keyword in keywords}
        ref.collect_papers(keywords)
        ref_prompt = ref.to_prompts(max_tokens=4096)

        prompt = f"Title: {title}\n References: {ref_prompt}\n Contributions: {cont}"
        output, _ = llm(systems=contribution_system_prompt_2, prompts=prompt, return_json=True)

        suggestions, _ = llm(systems=suggestions_system_prompt, prompts=str(output), return_json=True)

        return output, ref_prompt, suggestions
    else:
        ref = References(title=title, description=f"{contributions}")
        keywords, _ = llm(systems=SYSTEM["keywords"], prompts=title, return_json=True)
        keywords = {keyword: 10 for keyword in keywords}
        ref.collect_papers(keywords)
        ref_prompt = ref.to_prompts(max_tokens=4096)

        prompt = f"Title: {title}\n References: {ref_prompt}\n Contributions: {contributions}"
        output, _ = llm(systems=contribution_system_prompt_1, prompts=prompt, return_json=True)
        return output, ref_prompt, {}

def paste_title(suggestions):
    if suggestions:
        title = suggestions['title']['new title']
        contributions = suggestions['contributions']

        return title, contributions, {}, {}, {}
    else:
        return "", "", {}, {}, {}





with gr.Blocks() as demo:
    llm = GPTModel(model=default_model)
    gr.Markdown(ANNOUNCEMENT)

    with gr.Row():
        with gr.Column():
            title_input = gr.Textbox(label="Title")
            contribution_input = gr.Textbox(label="Contributions", lines=5)

            with gr.Row():
                button_reset = gr.Button("Reset")
                button_submit = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):
            contribution_output = gr.JSON(label="Contributions")
            suggestions_output = gr.JSON(label="Suggestions")
            button_copy = gr.Button("Send Title and Contributions to the Left")
            references_output = gr.JSON(label="References")

    button_reset.click(fn=reset, inputs=[], outputs=[title_input, contribution_input, contribution_output, references_output, suggestions_output])
    button_submit.click(fn=criticize_my_idea, inputs=[title_input, contribution_input, contribution_output, suggestions_output], outputs=[contribution_output, references_output, suggestions_output])
    button_copy.click(fn=paste_title, inputs=suggestions_output, outputs=[title_input, contribution_input, contribution_output, references_output, suggestions_output])
    # clear_button_refs.click(fn=clear_inputs_refs, inputs=[title_refs, slider_refs], outputs=[title_refs, slider_refs])
    # submit_button_refs.click(fn=wrapped_references_generator,
    #                          inputs=[title_refs, slider_refs, key], outputs=json_output)

demo.queue(concurrency_count=1, max_size=5, api_open=False)
demo.launch(show_error=True)
