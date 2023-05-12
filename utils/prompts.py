import logging
from langchain import PromptTemplate


log = logging.getLogger(__name__)


######################################################################################################################
# Some basic functions
######################################################################################################################
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



######################################################################################################################
# System Message
######################################################################################################################

# two parameters: min_refs_num, max_refs_num
keywords_system_template = """You are an assistant designed to provide accurate and informative keywords of searching academic papers.
Instructions
- Your response should always be a Python list; e.g. ["keyword1", "keyword2", "keyword3"]
- The length of list should between {min_refs_num} and {max_refs_num}
- Use specific phrases as keywords and avoid using too general words (e.g. machine learning)"""
# keywords_system_template = """You are an assistant designed to provide related research fields of academic papers.
# Instructions:
# - Your response should follow the following output format: ["field1", "field2", "field3"]\n
# - The length of this Python list should between {min_refs_num} and {max_refs_num}\n
# - Use specific phrases instead of using too general words (e.g. machine learning)"""

# two parameters: min_refs_num, max_refs_num
exp_methods_system_template = """You are an assistant designed to provide most related algorithms or methods to a given paper title.
Instructions
- Your response should always be a Python list; e.g. ["method_name_1", "method_name_2", "method_name_3"]
- The length of list should between {min_exps_num} and {max_exps_num}
- Use abbreviation to make each method's name have 5 characters or less."""

# one parameter: research_field
section_generation_system_template = r"""You are an assistant designed to write academic papers in the field of {research_field} using LaTeX. 
Instructions
- Your response should be professional and in academic tone.
- Always give a high-level overview at the beginning of each section or subsection. 
"""

KEYWORDS_SYSTEM = PromptTemplate(input_variables=["min_refs_num", "max_refs_num"],
                                 template=keywords_system_template)
EXP_METHODS_SYSTEM = PromptTemplate(input_variables=["min_exps_num", "max_exps_num"],
                                    template=exp_methods_system_template)
SECTION_GENERATION_SYSTEM = PromptTemplate(input_variables=["research_field"],
                                           template=section_generation_system_template)


######################################################################################################################
# Academic Paper
######################################################################################################################

INSTRUCTIONS = {"introduction":
                    "- Include five paragraph: Establishing the motivation for the research. Explaining its importance and relevance to the AI community. Clearly state the problem you're addressing, your proposed solution, and the specific research questions or objectives. Briefly mention key related works for context and explain the main differences from this work. List three novel contributions of this paper.",
               "results":
                    "Write the theoretical results section using LaTeX. Include theorem and corollary to support this paper (with formulas). Explain what assumptions are used and why they are standard and necessary. Do not include \section{...}. ",
                "conclusion":
                    "- Read the existing parts of paper and write the conclusion section.",
                "abstract":
                    "- Read the existing parts of paper and write the abstract."}


INSTRUCTIONS["backgrounds"] = "- Start from one high-level paragraph to state the central problem in this field with detailed examples in industrial applications and theoretical challenges. \n" \
                              "- Followed by two to three subsections:  Explain the foundational concepts and notations that underpin your research using as many as mathematical formulas (written in LaTeX). " \
                              "Introduce more necessary mathematical notations, equations, or algorithms that are connected to this work. Present detailed discussions on how these concepts are applied in this paper."


INSTRUCTIONS["related works"] = r"- Discuss three to five main related fields to this paper. " \
                                r"For each field, select five to ten key publications from references. " \
                                r"For each reference, analyze its strengths and weaknesses in one or two sentences. " \
                                r"Present the related works in a logical manner, often chronologically. " \
                                r"Consider using a taxonomy or categorization to structure the discussion. " \
                                r"Do not use \section{...} or \subsection{...}; use \paragraph{...} to list related fields. "

INSTRUCTIONS["methodology"] =  "- Provide a high-level overview of the proposed method at the beginning of this section. \n " \
                               "- Assume you have some figures ('fig1.png', 'fig2.png', ...); they can be any figures you need (e.g. flow chart, model architecture, sample output, simulation result, or others you need). Insert figures you need with informative caption. \n" \
                               "- Use one subsection to give a detailed formulation of the proposed method and explain how it overcomes the weakness of existing methods mentioned in this paper. " \
                                 " If necessary, write pseudo codes wrapped by \\begin{{algorithm}} ... \\end{{algorithm}} to explain the detailed steps instead of simply listing them. \n" \
                                "- Use one follow-up subsection to highlight the key concepts in the proposed method. " \
                                "  Elaborate the novelty of these key concepts using formulas and inserting appropriate figures. \n" \
                                "- Ensure the name of each subsection to be specific. \n"

INSTRUCTIONS["experiments"] =  "- Provide a high-level overview at the beginning of this section.\n " \
                               "- If necessary, include a table to compare with other methods and bold our method.\n" \
                               "- Assume you have some figures ('exp1.png', 'exp2.png', ...); they can be any figures you need (e.g. loss curves, comparison with other methods, visualization, or others you need). Insert figures you need with informative caption. \n" \
                               "- If necessary, use different subsections to distinguish different experimental setup."


def generate_paper_prompts(paper_info, section):
    title = paper_info["title"]
    description = paper_info["description"]
    references = paper_info["references"]
    paper = paper_info["body"]

    # fundamental_subprompt - describe the basic information of paper
    # instruction_subprompt - tell AI what to do
    # ref_instruction_subprompt - give AI references
    # self_subprompt - give AI existing written parts
    # output_subprompt - tell AI how to output
    fundamental_subprompt = "Your task is to write the {section} section of the machine learning paper with the title '{title}'. {description}\n"
    instruction_subprompt = "\n" \
                            "Your response should follow the following instructions:\n" \
                            "{instruction}\n" \
                            "- Start with \section{{{section}}}\n"
    ref_instruction_subprompt = "- Read references. " \
                                "Every time you use information from the references, you need to appropriately cite it (using \citep or \citet)." \
                                "For example of \citep, the sentence where you use information from lei2022adaptive \citep{{lei2022adaptive}}. " \
                                "For example of \citet, \citet{{lei2022adaptive}} claims some information.\n" \
                                "- Avoid citing the same reference in a same paragraph.\n" \
                                "\n" \
                                "References:\n" \
                                "{references}"
    self_subprompt = "The existing parts of this paper is provided here: {paper}.\n"
    output_subprompt = "Your response should start with \section{{{section}}}. Ensure that it can be directly compiled by LeTaX."
    abstract_output_subprompt  = "Your response should start with \\begin{{abstract}} and should end with \\end{{abstract}}. Ensure that it can be directly compiled by LeTaX."

    reivew_prompts =  PromptTemplate(
        input_variables=["title", "description", "instruction", "section", "references"],
        template=fundamental_subprompt + instruction_subprompt + ref_instruction_subprompt + output_subprompt)
    summarization_prompts = PromptTemplate(
        input_variables=["title", "description", "instruction", "section", "paper"],
        template=fundamental_subprompt + instruction_subprompt + self_subprompt + output_subprompt)
    abstract_prompts = PromptTemplate(
        input_variables=["title", "description", "instruction", "section", "paper"],
        template=fundamental_subprompt + instruction_subprompt + self_subprompt + abstract_output_subprompt)

    if section in ["introduction", "related works", "backgrounds"]:
        # title + references + instruction
        prompts = reivew_prompts.format(title=title,
                                          description=description,
                                          instruction=INSTRUCTIONS[section],
                                          section=section,
                                          references=references)
    elif section in ["abstract"]:
        # title + instruction + paper
        prompts = abstract_prompts.format(title=title,
                                          description=description,
                                          instruction=INSTRUCTIONS[section],
                                          section=section,
                                          paper=paper)
    elif section in ["methodology",  "experiments", "conclusion"]:
        # title + instruction + paper
        prompts = summarization_prompts.format(title=title,
                                          description=description,
                                          instruction=INSTRUCTIONS[section],
                                          section=section,
                                          paper=paper)
    else:
        raise NotImplementedError

    log.info(f"Generated prompts for {section}: {prompts}")
    return prompts


######################################################################################################################
# Literature Review
######################################################################################################################

BG_INSTRUCTIONS = {"introduction": "Please include four paragraph: Establishing the motivation for this survey. Explaining its importance and relevance to the AI community. Clearly state the coverage of this survey and the specific research questions or objectives. Briefly mention key related work for context. ",
                "related works": r"Please discuss key publications, methods, and techniques in related research area. Analyze the strengths and weaknesses of existing methods, and present the related works in a logical manner, often chronologically. Consider using a taxonomy or categorization to structure the discussion. Do not use \section{...} or \subsection{...}; use \paragraph{...} instead. ",
                "backgrounds": r"Please clearly state the central problem in this field. Explain the foundational theories, concepts, and principles that underpin your research using as many as mathematical formulas or equations (written in LaTeX). Introduce any necessary mathematical notations, equations, or algorithms that are central to this field (written them in LaTeX).  Do not include \section{...} but you can have \subsection{...}. ",}



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