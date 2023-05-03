from utils.references import References
from utils.prompts import generate_paper_prompts, generate_keywords_prompts, generate_experiments_prompts
from utils.gpt_interaction import get_responses, extract_responses, extract_keywords, extract_json
from utils.tex_processing import replace_title
from utils.figures import generate_random_figures
import datetime
import shutil
import time
import logging
import os

TOTAL_TOKENS = 0
TOTAL_PROMPTS_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0

def make_archive(source, destination):
    base = os.path.basename(destination)
    name = base.split('.')[0]
    format = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, format, archive_from, archive_to)
    shutil.move('%s.%s'%(name,format), destination)
    return destination


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
    prompts = generate_paper_prompts(paper, section)
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
    time.sleep(5)
    print(f"{section} has been generated. Saved to {tex_file}.")
    return usage



def generate_draft(title, description="", template="ICLR2022", model="gpt-4"):
    """
    The main pipeline of generating a paper.
        1. Copy everything to the output folder.
        2. Create references.
        3. Generate each section using `pipeline`.
        4. Post-processing: check common errors, fill the title, ...
    """
    paper = {}
    paper_body = {}

    # Create a copy in the outputs folder.
    # todo: use copy_templates function instead.
    now = datetime.datetime.now()
    target_name = now.strftime("outputs_%Y%m%d_%H%M%S")
    source_folder = f"latex_templates/{template}"
    destination_folder = f"outputs/{target_name}"
    shutil.copytree(source_folder, destination_folder)

    bibtex_path = destination_folder + "/ref.bib"
    save_to_path = destination_folder +"/"
    replace_title(save_to_path, title)
    logging.basicConfig( level=logging.INFO, filename=save_to_path+"generation.log")

    # Generate keywords and references
    print("Initialize the paper information ...")
    prompts = generate_keywords_prompts(title, description)
    gpt_response, usage = get_responses(prompts, model)
    keywords = extract_keywords(gpt_response)
    log_usage(usage, "keywords")
    ref = References(load_papers = "") #todo: allow users to upload bibfile.
    ref.collect_papers(keywords, method="arxiv") #todo: add more methods to find related papers
    all_paper_ids = ref.to_bibtex(bibtex_path) #todo: this will used to check if all citations are in this list

    print(f"The paper information has been initialized. References are saved to {bibtex_path}.")

    paper["title"] = title
    paper["description"] = description
    paper["references"] = ref.to_prompts() #todo: see if this prompts can be compressed.
    paper["body"] = paper_body
    paper["bibtex"] = bibtex_path

    print("Generating figures ...")
    prompts = generate_experiments_prompts(paper)
    gpt_response, usage = get_responses(prompts, model)
    list_of_methods = list(extract_json(gpt_response))
    log_usage(usage, "figures")
    generate_random_figures(list_of_methods, save_to_path + "comparison.png")

    for section in ["introduction", "related works", "backgrounds", "methodology", "experiments", "conclusion", "abstract"]:
        try:
            usage = pipeline(paper, section, save_to_path, model=model)
            log_usage(usage, section)
        except Exception as e:
            print(f"Failed to generate {section} due to the error: {e}")
    print(f"The paper {title} has been generated. Saved to {save_to_path}.")
    return make_archive(destination_folder, "output.zip")

if __name__ == "__main__":
    # title = "Training Adversarial Generative Neural Network with Adaptive Dropout Rate"
    title = "Playing Atari Game with Deep Reinforcement Learning"
    description = ""
    template = "ICLR2022"
    model = "gpt-4"
    # model = "gpt-3.5-turbo"

    generate_draft(title, description, template, model)
