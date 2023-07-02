import json
import os.path
from utils.references import References
from utils.knowledge import Knowledge
from utils.file_operations import hash_name, make_archive, copy_templates
from utils.tex_processing import create_copies
from section_generator import section_generation  # figures_generation, section_generation_bg, keywords_generation,
from utils.prompts import generate_paper_prompts
import logging
import time
from langchain.vectorstores import FAISS
from utils.gpt_interaction import GPTModel
from utils.prompts import SYSTEM
from models import EMBEDDINGS

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

    message = f">>USAGE>> For generating {generating_target}, {total_tokens} tokens have been used " \
              f"({prompts_tokens} for prompts; {completion_tokens} for completion). " \
              f"{TOTAL_TOKENS} tokens have been used in total."
    if print_out:
        print(message)
    logging.info(message)


def _generation_setup(title, description="", template="ICLR2022",
                      tldr=False, max_kw_refs=10, bib_refs=None, max_tokens_ref=2048,  # generating references
                      knowledge_database=None, max_tokens_kd=2048, query_counts=10,  # querying from knowledge database
                      debug=True):
    """
    This function handles the setup process for paper generation; it contains three folds
        1. Copy the template to the outputs folder. Create the log file `generation.log`
        2. Collect references based on the given `title` and `description`
        3. Generate the basic `paper` object (a dictionary)

    Parameters:
        title (str): The title of the paper.
        description (str, optional): A short description or abstract for the paper. Defaults to an empty string.
        template (str, optional): The template to be used for paper generation. Defaults to "ICLR2022".
        tldr (bool, optional): A flag indicating whether a TL;DR (Too Long; Didn't Read) summary should be used
                               for the collected papers. Defaults to False.
        max_kw_refs (int, optional): The maximum number of references that can be associated with each keyword.
                                     Defaults to 10.
        bib_refs (path to a bibtex file, optional).

    Returns:
    tuple: A tuple containing the following elements:
        - paper (dict): A dictionary containing the generated paper information.
        - destination_folder (str): The path to the destination folder where the generation log is saved.
        - all_paper_ids (list): A list of all paper IDs collected for the references.
    """
    # print("Generation setup...")
    # paper = {}
    # paper_body = {}
    llm = GPTModel(model="gpt-3.5-turbo")

    # Create a copy in the outputs folder.
    bibtex_path, destination_folder = copy_templates(template, title)
    logging.basicConfig(level=logging.INFO, filename=os.path.join(destination_folder, "generation.log"))

    ###################################################################################################################
    # Generate contributions
    ###################################################################################################################
    if description:
        contributions = description
    else:
        try:
            contributions, usage = llm(systems=SYSTEM["contributions"], prompts=title, return_json=True)
            contributions = [f"Contribution {idx}: {contributions[contribution]['statement']}\n" \
                             f"Novelty of Contribution {idx}: {contributions[contribution]['reason']}\n"
                             for idx, contribution in enumerate(contributions)]
            contributions = "".join(contributions)
            log_usage(usage, "contributions")
        except RuntimeError:
            if debug:
                raise RuntimeError("Failed to generate contributions.")
            else:
                print("Failed to generate contributions. Use empty contributions.")
                contributions = ""
    print("Contributions:\n{}".format(contributions))
    ###################################################################################################################
    # Generate references
    ###################################################################################################################
    # input_dict = {"title": title, "description": description}
    # keywords, usage = keywords_generation(input_dict)
    # log_usage(usage, "keywords")
    try:
        keywords, usage = llm(systems=SYSTEM["keywords"], prompts=title, return_json=True)
        log_usage(usage, "keywords")
        keywords = {keyword: max_kw_refs for keyword in keywords}
    except RuntimeError:
        if debug:
            raise RuntimeError("Failed to generate keywords.")
        else:
            print("Failed to generate keywords. Use default keywords.")
            keywords = {"machine learning": max_kw_refs, "artificial intelligence": max_kw_refs}  # DEFAULT KEYWORDS
    # generate keywords dictionary
    # keywords = {keyword: max_kw_refs for keyword in keywords}

    print("Keywords: \n", keywords)
    # todo: in some rare situations, collected papers will be an empty list. handle this issue
    ref = References(title, bib_refs)
    ref.collect_papers(keywords, tldr=tldr)
    references = ref.to_prompts(max_tokens=max_tokens_ref)
    all_paper_ids = ref.to_bibtex(bibtex_path)
    ###################################################################################################################
    # Generate domain knowledge
    ###################################################################################################################
    prompts = f"Title: {title}\n Contributions: {contributions}"
    preliminaries_kw, _ = llm(systems=SYSTEM["preliminaries"], prompts=prompts)
    # check if the database exists or not
    db_path = f"knowledge_databases/{knowledge_database}"
    db_config_path = os.path.join(db_path, "db_meta.json")
    db_index_path = os.path.join(db_path, "faiss_index")
    if os.path.isdir(db_path):
        try:
            # load configuration file
            with open(db_config_path, "r", encoding="utf-8") as f:
                db_config = json.load(f)
            model_name = db_config["embedding_model"]
            embeddings = EMBEDDINGS[model_name]
            db = FAISS.load_local(db_index_path, embeddings)
            knowledge = Knowledge(db=db)
            knowledge.collect_knowledge(preliminaries_kw, max_query=query_counts)
            domain_knowledge = knowledge.to_prompts(max_tokens_kd)
        except Exception as e:
            if debug:
                raise RuntimeError(f"Failed to query from FAISS. Error {e}.")
            else:
                print(f"Failed to query from FAISS. Error {e}. Use empty domain knowledge instead.")
                domain_knowledge = ""
    else:
        print("Selected database doesn't exist or no database is selected.")
        domain_knowledge = ""

    ###################################################################################################################
    # Generate necessary media
    ###################################################################################################################
    prompts = f"Title: {title}\n Contributions: {contributions}"
    try:
        components, usage = llm(systems=SYSTEM["components"], prompts=prompts, return_json=True)
        log_usage(usage, "media")
    except RuntimeError:
        if debug:
            raise RuntimeError("Failed to generate media.")
        else:
            print("Failed to generate media. Use default media.")
            components = {}

    print(f"The paper information has been initialized. References are saved to {bibtex_path}.")

    paper = {}
    paper_body = {}
    paper["title"] = title
    paper["description"] = contributions
    paper["references"] = references
    paper["body"] = paper_body
    paper["bibtex"] = bibtex_path
    paper["domain_knowledge"] = domain_knowledge
    paper["components"] = components

    # print(json.dumps(paper, indent=4))
    return paper, destination_folder, all_paper_ids
    # todo: use `all_paper_ids` to check if all citations are in this list


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


def generate_draft(title, description="", # main input
                   tldr=True, max_kw_refs=10, bib_refs=None, max_tokens_ref=2048,  # references
                   knowledge_database=None, max_tokens_kd=2048, query_counts=10, # domain knowledge
                   sections=None, model="gpt-4", template="ICLR2022", prompts_mode=False, # outputs parameters
                   ):
    """
    This function generates a draft paper using the provided information; it contains three steps: 1. Pre-processing:
    Initializes the setup for paper generation and filters the sections to be included in the paper. 2. Processing:
    Generates each section of the paper. 3. Post-processing: Creates backup copies of the paper and returns the paper
    in a zipped format.

    Parameters:
        title (str): The title of the paper.
        description (str, optional): A short description or abstract for the paper. Defaults to an empty string.
        template (str, optional): The template to be used for paper generation. Defaults to "ICLR2022".
        tldr (bool, optional): A flag indicating whether a TL;DR (Too Long; Didn't Read) summary should be used
                               for the collected papers. Defaults to True.
        max_kw_refs (int, optional): The maximum number of references that can be associated with each keyword.
                                     Defaults to 10.
        sections (list, optional): The sections to be included in the paper. If not provided, all the standard
                                   sections are included.
        bib_refs (path to a bibtex file, optional).
        model (str, optional): The language model to be used for paper generation. Defaults to "gpt-4".

    Returns:
    str: The path to the zipped file containing the generated paper and associated files.

    Note: The function also handles errors that occur during section generation and retries a maximum of 4 times
    before proceeding.
    """

    def _filter_sections(sections):
        ordered_sections = ["introduction", "related works", "backgrounds", "methodology", "experiments", "conclusion",
                            "abstract"]
        return [section for section in ordered_sections if section in sections]

    # pre-processing `sections` parameter;
    print("================START================")
    print(f"Generating the paper '{title}'.")
    print("================PRE-PROCESSING================")
    # make `sections` in a correct order
    if sections is None:
        sections = ["introduction", "related works", "backgrounds", "methodology", "experiments", "conclusion",
                    "abstract"]
    else:
        sections = _filter_sections(sections)
    paper, destination_folder, _ = _generation_setup(title, description, template, tldr, max_kw_refs, bib_refs,
                                                     max_tokens_ref=max_tokens_ref, max_tokens_kd=max_tokens_kd,
                                                     query_counts=query_counts,
                                                     knowledge_database=knowledge_database)

    # main components
    prompts_dict = {}
    print(f"================PROCESSING================")
    for section in sections:
        if prompts_mode:
            prompts = generate_paper_prompts(paper, section)
            prompts_dict[section] = prompts
            continue

        print(f"Generate {section} part...")
        max_attempts = 4
        attempts_count = 0
        while attempts_count < max_attempts:
            try:
                usage = section_generation(paper, section, destination_folder, model=model)
                print(f"{section} part has been generated. ")
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

    if prompts_mode:
        filename = hash_name(input_dict) + ".json"
        with open(filename, "w") as f:
            json.dump(prompts_dict, f)
        return filename
    else:
        return make_archive(destination_folder, filename)


if __name__ == "__main__":
    import openai

    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.api_base = os.getenv("OPENAI_API_BASE")

    target_title = "Playing Atari with Decentralized Reinforcement Learning"
    output = generate_draft(target_title, knowledge_database="ml_textbook_test")
    print(output)
