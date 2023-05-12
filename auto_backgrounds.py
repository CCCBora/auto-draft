import os.path
import json
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
              f"{TOTAL_TOKENS} tokens have been used in total.\n\n"
    if print_out:
        print(message)
    logging.info(message)

def _generation_setup(title, description="", template="ICLR2022", model="gpt-4",
                      tldr=False, max_kw_refs=4, max_num_refs=10):
    print("Generation setup...")
    paper = {}
    paper_body = {}

    # Create a copy in the outputs folder.
    bibtex_path, destination_folder = copy_templates(template, title)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(destination_folder, "generation.log") )

    # Generate keywords and references
    print("Initialize the paper information ...")
    input_dict = {"title": title, "description": description}
    # keywords, usage = keywords_generation(input_dict, model="gpt-3.5-turbo", max_kw_refs=max_kw_refs)
    keywords, usage = keywords_generation(input_dict) #todo: handle format error here
    print(f"keywords: {keywords}")
    log_usage(usage, "keywords")

    # generate keywords dictionary
    keywords = {keyword:max_kw_refs for keyword in keywords}
    # tmp = {}
    # for keyword in json.loads(keywords):
    #     tmp[keyword] = max_kw_refs
    # keywords = tmp
    print(f"keywords: {keywords}")

    ref = References()
    ref.collect_papers(keywords, tldr=tldr)
    # todo: use `all_paper_ids` to check if all citations are in this list
    #       in tex_processing, remove all duplicated ids
    #       find most relevant papers; max_num_refs
    all_paper_ids = ref.to_bibtex(bibtex_path)

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


def generate_draft(title, description="", template="ICLR2022", model="gpt-4", tldr=True, max_kw_refs=4):
    paper, destination_folder, _ = _generation_setup(title, description, template, model, tldr, max_kw_refs)
    raise
    # todo: `list_of_methods` failed to be generated; find a solution ...
    # print("Generating figures ...")
    # usage = figures_generation(paper, destination_folder, model="gpt-3.5-turbo")
    # log_usage(usage, "figures")

    # for section in ["introduction", "related works", "backgrounds", "methodology", "experiments", "conclusion", "abstract"]:
    for section in ["introduction", "related works", "backgrounds", "methodology", "experiments", "conclusion", "abstract"]:
        max_attempts = 4
        attempts_count = 0
        while attempts_count < max_attempts:
            try:
                usage = section_generation(paper, section, destination_folder, model=model)
                log_usage(usage, section)
                break
            except Exception as e:
                message = f"Failed to generate {section}. {type(e).__name__} was raised:  {e}"
                print(message)
                logging.info(message)
                attempts_count += 1
                time.sleep(20)

    input_dict = {"title": title, "description": description, "generator": "generate_draft"}
    filename = hash_name(input_dict) + ".zip"
    return make_archive(destination_folder, filename)


if __name__ == "__main__":
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    title = "Using interpretable boosting algorithms for modeling environmental and agricultural data"
    description = ""
    output = generate_draft(title, description, tldr=True, max_kw_refs=10)
    print(output)