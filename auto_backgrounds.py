from utils.references import References
from utils.file_operations import hash_name, make_archive, copy_templates
from section_generator import section_generation_bg, keywords_generation, figures_generation, section_generation
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
              f"{TOTAL_TOKENS} tokens have been used in total."
    if print_out:
        print(message)
    logging.info(message)

def _generation_setup(title, description="", template="ICLR2022", model="gpt-4"):
    '''
    todo: use `model` to control which model to use; may use another method to generate keywords or collect references
    '''
    paper = {}
    paper_body = {}

    # Create a copy in the outputs folder.
    bibtex_path, destination_folder = copy_templates(template, title)
    logging.basicConfig(level=logging.INFO, filename=destination_folder + "/generation.log")

    # Generate keywords and references
    print("Initialize the paper information ...")
    input_dict = {"title": title, "description": description}
    keywords, usage = keywords_generation(input_dict, model="gpt-3.5-turbo")
    print(f"keywords: {keywords}")
    log_usage(usage, "keywords")

    ref = References(load_papers="")
    ref.collect_papers(keywords, method="arxiv")
    all_paper_ids = ref.to_bibtex(bibtex_path)  # todo: this will used to check if all citations are in this list

    print(f"The paper information has been initialized. References are saved to {bibtex_path}.")

    paper["title"] = title
    paper["description"] = description
    paper["references"] = ref.to_prompts()
    paper["body"] = paper_body
    paper["bibtex"] = bibtex_path
    return paper, destination_folder, all_paper_ids



def generate_backgrounds(title, description="", template="ICLR2022", model="gpt-4"):
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


def fake_generator(title, description="", template="ICLR2022", model="gpt-4"):
    """
    This function is used to test the whole pipeline without calling OpenAI API.
    """
    input_dict = {"title": title, "description": description, "generator": "generate_draft"}
    filename = hash_name(input_dict) + ".zip"
    return make_archive("sample-output.pdf", filename)


def generate_draft(title, description="", template="ICLR2022", model="gpt-4"):
    paper, destination_folder, _ = _generation_setup(title, description, template, model)

    # todo: `list_of_methods` failed to be generated; find a solution ...
    # print("Generating figures ...")
    # usage = figures_generation(paper, destination_folder, model="gpt-3.5-turbo")
    # log_usage(usage, "figures")

    # for section in ["introduction", "related works", "backgrounds", "methodology", "experiments", "conclusion", "abstract"]:
    for section in ["introduction", "related works", "backgrounds", "abstract"]:
        try:
            usage = section_generation(paper, section, destination_folder, model=model)
            log_usage(usage, section)
        except Exception as e:
            message = f"Failed to generate {section}. {type(e).__name__} was raised:  {e}"
            print(message)
            logging.info(message)
            max_attempts = 2
            # re-try `max_attempts` time
            for i in range(max_attempts):
                time.sleep(20)
                try:
                    usage = section_generation(paper, section, destination_folder, model=model)
                    log_usage(usage, section)
                    e = None
                except Exception as e:
                    pass
                if e is None:
                    break


    input_dict = {"title": title, "description": description, "generator": "generate_draft"}
    filename = hash_name(input_dict) + ".zip"
    return make_archive(destination_folder, filename)
