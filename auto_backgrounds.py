import os.path
from utils.references import References
from utils.file_operations import hash_name, make_archive, copy_templates
from utils.tex_processing import create_copies
from section_generator import section_generation_bg, keywords_generation, figures_generation, section_generation
from references_generator import generate_top_k_references
import logging
import time


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
              f"{TOTAL_TOKENS} tokens have been used in total.\n\n"
    if print_out:
        print(message)
    logging.info(message)

def _generation_setup(title, description="", template="ICLR2022", tldr=False,
                      max_kw_refs=10, max_num_refs=50, bib_refs=None, max_tokens=2048):
    """
    This function handles the setup process for paper generation; it contains three folds
        1. Copy the template to the outputs folder. Create the log file `generation.log`
        2. Collect references based on the given `title` and `description`
        3. Generate the basic `paper` object (a dictionary)

    Parameters:
        title (str): The title of the paper.
        description (str, optional): A short description or abstract for the paper. Defaults to an empty string.
        template (str, optional): The template to be used for paper generation. Defaults to "ICLR2022".
        tldr (bool, optional): A flag indicating whether a TL;DR (Too Long; Didn't Read) summary should be generated for the collected papers. Defaults to False.
        max_kw_refs (int, optional): The maximum number of references that can be associated with each keyword. Defaults to 10.
        max_num_refs (int, optional): The maximum number of references that can be included in the paper. Defaults to 50.
        bib_refs (list, optional): A list of pre-existing references in BibTeX format. Defaults to None.

    Returns:
    tuple: A tuple containing the following elements:
        - paper (dict): A dictionary containing the generated paper information.
        - destination_folder (str): The path to the destination folder where the generation log is saved.
        - all_paper_ids (list): A list of all paper IDs collected for the references.
    """
    # print("Generation setup...")
    paper = {}
    paper_body = {}

    # Create a copy in the outputs folder.
    bibtex_path, destination_folder = copy_templates(template, title)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(destination_folder, "generation.log") )

    # Generate keywords and references
    # print("Initialize the paper information ...")
    input_dict = {"title": title, "description": description}
    keywords, usage = keywords_generation(input_dict)
    log_usage(usage, "keywords")

    # generate keywords dictionary
    keywords = {keyword:max_kw_refs for keyword in keywords}
    print(f"keywords: {keywords}\n\n")

    ref = References(title, bib_refs)
    ref.collect_papers(keywords, tldr=tldr)
    all_paper_ids = ref.to_bibtex(bibtex_path)

    print(f"The paper information has been initialized. References are saved to {bibtex_path}.")

    paper["title"] = title
    paper["description"] = description
    paper["references"] = ref.to_prompts(max_tokens=max_tokens)
    paper["body"] = paper_body
    paper["bibtex"] = bibtex_path
    return paper, destination_folder, all_paper_ids #todo: use `all_paper_ids` to check if all citations are in this list



def generate_backgrounds(title, description="", template="ICLR2022", model="gpt-4"):
    # todo: to match the current generation setup
    paper, destination_folder, _ = _generation_setup(title, description, template, model)

    for section in ["introduction", "related works", "backgrounds"]:
        try:
            usage = section_generation_bg(paper, section, destination_folder, model=model)
            log_usage(usage, section)
        except Exception as e:
            message = f"Failed to generate {section}. {type(e).__name__} was raised:  {e}"
            print(message)
            logging.info(message)
    print(f"The paper '{title}' has been generated. Saved to {destination_folder}.")

    input_dict = {"title": title, "description": description, "generator": "generate_backgrounds"}
    filename = hash_name(input_dict) + ".zip"
    return make_archive(destination_folder, filename)



def generate_draft(title, description="", template="ICLR2022",
                   tldr=True, max_kw_refs=10, max_num_refs=30, sections=None, bib_refs=None, model="gpt-4"):
    # pre-processing `sections` parameter;
    print("================PRE-PROCESSING================")
    if sections is None:
        sections = ["introduction", "related works", "backgrounds", "methodology", "experiments", "conclusion", "abstract"]

    # todo: add more parameters; select which section to generate; select maximum refs.
    paper, destination_folder, _ = _generation_setup(title, description, template, tldr, max_kw_refs, max_num_refs, bib_refs)

    # main components
    for section in sections:
        print(f"================Generate {section}================")
        max_attempts = 4
        attempts_count = 0
        while attempts_count < max_attempts:
            try:
                usage = section_generation(paper, section, destination_folder, model=model)
                log_usage(usage, section)
                break
            except Exception as e:
                message = f"Failed to generate {section}. {type(e).__name__} was raised:  {e}\n"
                print(message)
                logging.info(message)
                attempts_count += 1
                time.sleep(15)

    # post-processing
    print("================POST-PROCESSING================")
    create_copies(destination_folder)
    input_dict = {"title": title, "description": description, "generator": "generate_draft"}
    filename = hash_name(input_dict) + ".zip"
    print("\nMission completed.\n")
    return make_archive(destination_folder, filename)







if __name__ == "__main__":
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    title = "Using interpretable boosting algorithms for modeling environmental and agricultural data"
    description = ""
    output = generate_draft(title, description, tldr=True, max_kw_refs=10)
    print(output)