from utils.references import References
from utils.prompts import generate_bg_keywords_prompts, generate_bg_summary_prompts
from utils.gpt_interaction import get_responses, extract_responses, extract_keywords, extract_json
from utils.tex_processing import replace_title
import datetime
import shutil
import time
import logging

TOTAL_TOKENS = 0
TOTAL_PROMPTS_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0


def log_usage(usage, generating_target, print_out=True):
    global TOTAL_TOKENS
    global TOTAL_PROMPTS_TOKENS
    global TOTAL_COMPLETION_TOKENS

    prompts_tokens = usage['prompt_tokens']
    completion_tokens = usage['completion_tokens']
    total_tokens = usage['total_tokens']

    TOTAL_TOKENS += total_tokens
    TOTAL_PROMPTS_TOKENS += prompts_tokens
    TOTAL_COMPLETION_TOKENS += completion_tokens

    message = f"For generating {generating_target}, {total_tokens} tokens have been used ({prompts_tokens} for prompts; {completion_tokens} for completion). " \
              f"{TOTAL_TOKENS} tokens have been used in total."
    if print_out:
        print(message)
    logging.info(message)

def pipeline(paper, section, save_to_path, model):
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
    tex_file = save_to_path + f"{section}.tex"
    if section == "abstract":
        with open(tex_file, "w") as f:
            f.write(r"\begin{abstract}")
        with open(tex_file, "a") as f:
            f.write(output)
        with open(tex_file, "a") as f:
            f.write(r"\end{abstract}")
    else:
        with open(tex_file, "w") as f:
            f.write(f"\section{{{section}}}\n")
        with open(tex_file, "a") as f:
            f.write(output)
    time.sleep(20)
    print(f"{section} has been generated. Saved to {tex_file}.")
    return usage



def generate_backgrounds(title, description="", template="ICLR2022", model="gpt-4"):
    paper = {}
    paper_body = {}

    # Create a copy in the outputs folder.
    now = datetime.datetime.now()
    target_name = now.strftime("outputs_%Y%m%d_%H%M%S")
    source_folder = f"latex_templates/{template}"
    destination_folder = f"outputs/{target_name}"
    shutil.copytree(source_folder, destination_folder)

    bibtex_path = destination_folder + "/ref.bib"
    save_to_path = destination_folder +"/"
    replace_title(save_to_path, "A Survey on " + title)
    logging.basicConfig( level=logging.INFO, filename=save_to_path+"generation.log")

    # Generate keywords and references
    print("Initialize the paper information ...")
    prompts = generate_bg_keywords_prompts(title, description)
    gpt_response, usage = get_responses(prompts, model)
    keywords = extract_keywords(gpt_response)
    log_usage(usage, "keywords")

    ref = References(load_papers = "")
    ref.collect_papers(keywords, method="arxiv")
    all_paper_ids = ref.to_bibtex(bibtex_path) #todo: this will used to check if all citations are in this list

    print(f"The paper information has been initialized. References are saved to {bibtex_path}.")

    paper["title"] = title
    paper["description"] = description
    paper["references"] = ref.to_prompts() # to_prompts(top_papers)
    paper["body"] = paper_body
    paper["bibtex"] = bibtex_path

    for section in ["introduction", "related works", "backgrounds"]:
        try:
            usage = pipeline(paper, section, save_to_path, model=model)
            log_usage(usage, section)
        except Exception as e:
            print(f"Failed to generate {section} due to the error: {e}")
    print(f"The paper {title} has been generated. Saved to {save_to_path}.")

if __name__ == "__main__":
    title = "Reinforcement Learning"
    description = ""
    template = "Summary"
    model = "gpt-4"
    # model = "gpt-3.5-turbo"

    generate_backgrounds(title, description, template, model)
