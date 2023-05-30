from utils.prompts import generate_paper_prompts, generate_keywords_prompts, generate_experiments_prompts, generate_bg_summary_prompts
from utils.gpt_interaction import get_responses, extract_responses, extract_keywords, extract_json
from utils.figures import generate_random_figures
import time
import os
from utils.prompts import KEYWORDS_SYSTEM, SECTION_GENERATION_SYSTEM
from utils.gpt_interaction import get_gpt_responses
import json

#  three GPT-based content generator:
#       1. section_generation: used to generate main content of the paper
#       2. keywords_generation: used to generate a json output {key1: output1, key2: output2} for multiple purpose.
#       3. figure_generation: used to generate sample figures.
#  all generator should return the token usage.

MAX_ATTEMPTS = 6

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
    output = gpt_response # extract_responses(gpt_response)
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


def section_generation(paper, section, save_to_path, model, research_field="machine learning"):
    """
    The main pipeline of generating a section.
        1. Generate prompts.
        2. Get responses from AI assistant.
        3. Extract the section text.
        4. Save the text to .tex file.
    :return usage
    """
    prompts = generate_paper_prompts(paper, section)
    output, usage= get_gpt_responses(SECTION_GENERATION_SYSTEM.format(research_field=research_field), prompts,
                             model=model, temperature=0.4)
    paper["body"][section] = output
    tex_file = os.path.join(save_to_path, f"{section}.tex")
    with open(tex_file, "w") as f:
        f.write(output)
    time.sleep(5)
    return usage


def keywords_generation(input_dict, default_keywords=None):
    '''
    Input:
        input_dict: a dictionary containing the title of a paper.
        default_keywords: if anything went wrong, return this keywords.

    Output:
        a dictionary including all keywords and their importance score.

    Input example: {"title": "The title of a Machine Learning Paper"}
    Output Example: {"machine learning": 5, "reinforcement learning": 2}
    '''
    title = input_dict.get("title")
    attempts_count = 0
    while (attempts_count < MAX_ATTEMPTS) and (title is not None):
        try:
            keywords, usage= get_gpt_responses(KEYWORDS_SYSTEM.format(min_refs_num=1, max_refs_num=10), title,
                                     model="gpt-3.5-turbo", temperature=0.4)
            print(keywords)
            output = json.loads(keywords)
            return output.keys(), usage
        except json.decoder.JSONDecodeError:
            attempts_count += 1
            time.sleep(10)
    # Default references
    print("Error: Keywords generation has failed. Return the default keywords.")
    if default_keywords is None or isinstance(default_keywords, dict):
        return {"machine learning": 10}
    else:
        return default_keywords

def figures_generation(paper, save_to_path, model):
    # todo: this function is not complete.
    prompts = generate_experiments_prompts(paper)
    gpt_response, usage = get_responses(prompts, model)
    list_of_methods = list(extract_json(gpt_response))
    generate_random_figures(list_of_methods, os.path.join(save_to_path, "comparison.png"))
    return usage