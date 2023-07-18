from langchain.prompts import load_prompt
import os

def generate_paper_prompts(paper, section_name):
    section_name = section_name.replace(" ", "_")
    try:
        cur_path = os.path.dirname(__file__)
        target_path = os.path.join(cur_path, f"{section_name}.yaml")
        prompt = load_prompt(target_path)
    except FileNotFoundError:
        raise ValueError(f"Cannot find the prompt for the section name {section_name}. Please check the folder `prompts`.")
    kw = {k: paper[k] for k in prompt.input_variables}
    return prompt.format(**kw)