from utils.prompts import generate_paper_prompts, generate_keywords_prompts, generate_experiments_prompts, generate_bg_summary_prompts
from utils.gpt_interaction import get_responses, extract_responses, extract_keywords, extract_json
import time
import os

#  three GPT-based content generator:
#       1. section_generation: used to generate main content of the paper
#       2. keywords_generation: used to generate a json output {key1: output1, key2: output2} for multiple purpose.
#       3. figure_generation: used to generate sample figures.
#  all generator should return the token usage.


def section_generation_bg(paper, section, save_to_path, model):
    """
    The main pipeline of generating a section.
        1. Generate prompts.
        2. Get responses from AI assistant.
        3. Extract the section text.
        4. Save the text to .tex file.
    :return usage
    """
    print(f"Generating {section}...")
    prompts = generate_bg_summary_prompts(paper, section)
    gpt_response, usage = get_responses(prompts, model)
    output = extract_responses(gpt_response)
    paper["body"][section] = output
    tex_file = os.path.join(save_to_path, f"{section}.tex")
    # tex_file = save_to_path + f"/{section}.tex"
    if section == "abstract":
        with open(tex_file, "w") as f:
            f.write(r"\begin{abstract}")
        with open(tex_file, "a") as f:
            f.write(output)
        with open(tex_file, "a") as f:
            f.write(r"\end{abstract}")
    else:
        with open(tex_file, "w") as f:
            f.write(f"\section{{{section.upper()}}}\n")
        with open(tex_file, "a") as f:
            f.write(output)
    time.sleep(5)
    print(f"{section} has been generated. Saved to {tex_file}.")
    return usage


def keywords_generation(input_dict,  model):
    title = input_dict.get("title")
    description = input_dict.get("description", "")
    if title is not None:
        prompts = generate_keywords_prompts(title, description)
        gpt_response, usage = get_responses(prompts, model)
        keywords = extract_keywords(gpt_response)
        return keywords, usage
    else:
        raise ValueError("`input_dict` must include the key 'title'.")

def figures_generation():
    pass