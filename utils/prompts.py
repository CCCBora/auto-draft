import logging
log = logging.getLogger(__name__)

INSTRUCTIONS = {"introduction": "Please include five paragraph: Establishing the motivation for the research. Explaining its importance and relevance to the AI community. Clearly state the problem you're addressing, your proposed solution, and the specific research questions or objectives. Briefly mention key related work for context. Explain the main differences from your work. ",
                "related works": r"Please discuss key publications, methods, and techniques in your research area. Analyze the strengths and weaknesses of existing methods, and present the related works in a logical manner, often chronologically. Consider using a taxonomy or categorization to structure the discussion. Do not use \section{...} or \subsection{...}; use \paragraph{...} instead. ",
                "backgrounds": r"Please clearly state the central problem in this field. Explain the foundational theories, concepts, and principles that underpin your research using as many as mathematical formulas or equations (written in LaTeX). Introduce any necessary mathematical notations, equations, or algorithms that are central to this field (written them in LaTeX).  Do not include \section{...} but you can have \subsection{...}. ",
                "methodology": "Please read the paper I have written and write the methodology section with three subsections: Concisely describe the techniques, algorithms, and procedures employed to address the research problem (use as many as formulas written in LaTeX). Explain the rationale behind choosing these methods, and provide sufficient detail for replication (use as many as formulas written in LaTeX). Do not make any list steps; instead, just put them in the same paragraph with sufficient explainations. Do not include \section{...} but you can have \subsection{...}. ",
                "results": "Please write the theoretical results section using LaTeX. Include theorem and corollary to support this paper (with formulas). Explain what assumptions are used and why they are standard and necessary. Do not include \section{...}. ",
                "experiments": "Please write the experiment section using LaTeX. Include a table to compare with other methods and bold our method. Include one figure comparison.png; this figure compares the loss curve with other methods. Do not include \section{...}. ",
                "conclusion": "Please read the paper I have written and write the conclusion section.",
                "abstract": "Please read the paper I have written and write the abstract."}

BG_INSTRUCTIONS = {"introduction": "Please include four paragraph: Establishing the motivation for this survey. Explaining its importance and relevance to the AI community. Clearly state the coverage of this survey and the specific research questions or objectives. Briefly mention key related work for context. ",
                "related works": r"Please discuss key publications, methods, and techniques in related research area. Analyze the strengths and weaknesses of existing methods, and present the related works in a logical manner, often chronologically. Consider using a taxonomy or categorization to structure the discussion. Do not use \section{...} or \subsection{...}; use \paragraph{...} instead. ",
                "backgrounds": r"Please clearly state the central problem in this field. Explain the foundational theories, concepts, and principles that underpin your research using as many as mathematical formulas or equations (written in LaTeX). Introduce any necessary mathematical notations, equations, or algorithms that are central to this field (written them in LaTeX).  Do not include \section{...} but you can have \subsection{...}. ",}

def generate_keywords_prompts(title, description="", num_refs=5):
    prompts = f"I am writing a machine learning paper with the title '{title}'. {description}\n" \
                f"Generate three to five keywords. For each keyword, rate it from 1 to {num_refs}; the larger number means more important." \
                r"Your response must be in JSON format like {\"keyword1\":1, \"keyword2\":3}."
    return prompts

def generate_rename_prompts(paper_info, section):
    prompts = f"Please read the {section} section of the paper {paper_info['title']}: {paper_info['body'][section]}. \n" \
              f"You need to rename this section to make it more specific to the context. " \
              r"Response in a dictionary format like {\"option_1\": \"new_section_name_1\", \"option_2\": \"new_section_name_2\", ...}."
    return prompts

def generate_experiments_prompts(paper_info):
    prompts = f"I am writing a machine learning paper with the title {paper_info['title']}\n" \
              f"Please list two to four methods that I should compare my methods with and assign them with scores (5 means most related, 1 means least related). " \
              r"Response in a dictionary format like {\"method_name_1\": 2, \"method_name_2\": 5, ...}. Use abbreviation to make their names have 5 characters or less."
    return prompts



def generate_paper_prompts(paper_info, section):
    title = paper_info["title"]
    description = paper_info["description"]
    references = paper_info["references"]
    paper = paper_info["body"]

    # fundamental_subprompt - describe the basic information of paper
    # instruction_subprompt - tell AI what to do
    # references_subprompt - give AI references
    # self_subprompt - give AI existing written parts
    # output_subprompt - tell AI how to output

    fundamental_subprompt = f"I am writing a machine learning paper with the title '{title}'. {description}\n"
    instruction_subprompt = f"You need to write the {section} section. {INSTRUCTIONS[section]}\n"
    references_subprompt = f"Please read the following references: \n{references}\n"\
                            f"Every time you use information from the references, you need to cite its id after the sentence; " \
                           f"for example, the sentence where you use information from 1905.09788 \cite{{1905.09788}}. " \
                           f"Please avoid citing the same reference in the same paragraph. \n"
    self_subprompt = f"Here is the paper that I have written: {paper}.\n"
    output_subprompt = r"Put your response (do not include \section{...}) in the following Python script:" \
                        f"with open(\"{section}.tex\", \"w\") as f: f.write(r'''your_response''')"

    if section in ["introduction", "related works", "backgrounds"]:
        # title + references + instruction
        prompts = fundamental_subprompt + instruction_subprompt + references_subprompt + output_subprompt
    elif section in ["experiments"]:
        # only title and instruction
        prompts = fundamental_subprompt + instruction_subprompt + output_subprompt
    elif section in ["methodology", "abstract", "conclusion"]:
        # title + instruction + paper
        prompts = fundamental_subprompt + instruction_subprompt + self_subprompt + output_subprompt
    else:
        raise NotImplementedError

    log.info(f"Generated prompts for {section}: {prompts}")
    return prompts


def generate_bg_summary_prompts(paper_info, section):
    title = paper_info["title"]
    description = paper_info["description"]
    references = paper_info["references"]
    paper = paper_info["body"]

    # fundamental_subprompt - describe the basic information of paper
    # instruction_subprompt - tell AI what to do
    # references_subprompt - give AI references
    # self_subprompt - give AI existing written parts
    # output_subprompt - tell AI how to output

    fundamental_subprompt = f"I am writing a machine learning survey about '{title}'. {description}\n"
    instruction_subprompt = f"You need to write the {section} section. {INSTRUCTIONS[section]}\n"
    references_subprompt = f"Please read the following references: \n{references}\n"\
                            f"Every time you use information from the references, you need to cite its id after the sentence; " \
                           f"for example, the sentence where you use information from 1905.09788 \cite{{1905.09788}}. " \
                           f"Please avoid citing the same reference in the same paragraph. \n"
    self_subprompt = f"Here is the paper that I have written: {paper}.\n"
    output_subprompt = r"Put your response (do not include \section{...}) in the following Python script:" \
                        f"with open(\"{section}.tex\", \"w\") as f: f.write(r'''your_response''')"

    if section in ["introduction", "related works", "backgrounds"]:
        # title + references + instruction
        prompts = fundamental_subprompt + instruction_subprompt + references_subprompt + output_subprompt
    else:
        raise NotImplementedError

    log.info(f"Generated prompts for {section}: {prompts}")
    return prompts