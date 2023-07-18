import os
import openai
from utils.references import References
from utils.gpt_interaction import GPTModel
from prompts import SYSTEM
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type

MAX_TOKENS = 2048

openai.api_key = os.getenv("OPENAI_API_KEY")
default_model = os.getenv("DEFAULT_MODEL")
if default_model is None:
    default_model = "gpt-3.5-turbo-16k"
llm = GPTModel(model=default_model, delay=1)

paper_system_prompt =  '''You are an assistant designed to propose choices of research direction.
The user will input questions or some keywords of a fields. You need to generate some paper titles and main contributions. Ensure follow the following instructions:
Instruction:
- Your response should follow the JSON format.
- Your response should have the following structure: 
{
    "your suggested paper title": 
        {
            "summary": "an overview introducing what this paper will include",
            "contributions": {
                "contribution1": {"statement": "briefly describe this contribution", "reason": "reason why this contribution can make this paper outstanding"}, 
                "contribution2": {"statement": "briefly describe this contribution", "reason": "reason why this contribution can make this paper outstanding"}, 
                ...
            }
        }
    "your suggested paper title": 
        {
            "summary": "an overview introducing what this paper will include",
            "contributions": {
                "contribution1": {"statement": "briefly describe this contribution", "reason": "reason why this contribution can make this paper outstanding"}, 
                "contribution2": {"statement": "briefly describe this contribution", "reason": "reason why this contribution can make this paper outstanding"}, 
                ...
            }
        }
    ...
}
- Please list three to five suggested title and at least three contributions for each paper.
'''


contribution_system_prompt = '''You are an assistant designed to criticize the contributions of a paper. You will be provided Paper's Title, References and Contributions. Ensure follow the following instructions:
Instruction:
- Your response should follow the JSON format.
- Your response should have the following structure: 
{
    "title": "the title provided by the user",
    "comment": "your thoughts on if this title clearly reflects the key ideas of this paper and explain why"
    "contributions": {
        "contribution1": {"statement": "briefly describe what the contribution is", 
                          "reason": "reason why the user claims it is a contribution", 
                          "judge": "your thought about if this is a novel contribution and explain why", 
                          "suggestion": "your suggestion on how to modify the research direction to enhance the novelty "},
        "contribution2": {"statement": "briefly describe what the contribution is", 
                          "reason": "reason why the user claims it is a contribution", 
                          "judge": "your thought about if this is a novel contribution and explain why", 
                          "suggestion": "your suggestion on how to modify the research direction to enhance the novelty "},
        ...
    }
} 
- You need to carefully check if the claimed contribution has been made in the provided references, which makes the contribution not novel.  
- You also need to propose your concerns on if any of contributions could be incremental or just a mild modification on an existing work.
'''


def find_research_directions(research_field):
    output, _ = llm(systems=paper_system_prompt, prompts=research_field, return_json=False)
    return output

def find_references(title, contributions):
    max_tokens = MAX_TOKENS
    ref = References(title=title, description=f"{contributions}")
    keywords, _ = llm(systems=SYSTEM["keywords"], prompts=title, return_json=True)
    keywords = {keyword: 10 for keyword in keywords}
    ref.collect_papers(keywords)
    ref_prompt = ref.to_prompts(max_tokens=max_tokens)
    return ref_prompt


def judge_novelty(title, contributions):
    max_tokens = MAX_TOKENS
    ref = References(title=title, description=f"{contributions}")
    keywords, _ = llm(systems=SYSTEM["keywords"], prompts=title, return_json=True)
    keywords = {keyword: 10 for keyword in keywords}
    ref.collect_papers(keywords)
    ref_prompt = ref.to_prompts(max_tokens=max_tokens)
    prompt = f"Title: {title}\n References: {ref_prompt}\n Contributions: {contributions}"
    output, _ = llm(systems=contribution_system_prompt, prompts=prompt, return_json=False)
    return output


functions = [
    {
        "name": "find_research_directions",
        "description": "when your student has already shown interests in a specific topic and provided a rough description of potential contributions, help your student to dive this direction deeper",
        "parameters": {
            "type": "object",
            "properties": {
                "research_description": {
                    "type": "string",
                    "description": "a paragraph with details in English describing "
                                   "(1) what is the main problem you are trying to solve "
                                   "(2) what is the main novelty of this idea (3) how to complete this research."
                }
            },
            "required": ["research_description"],
        },
    },
    {
        "name": "find_references",
        "description": "find references for given details of a paper",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "the title (in English) of the academic paper your student will write.",
                },
                "contributions": {"type": "string",
                         "description": "a general description on the contributions of this paper in English."
                                                "If there are multiple contributions, index them with numbers."},
            },
            "required": ["title", "contributions"],
        },
    },
    {
        "name": "judge_novelty",
        "description": "evaluate the novelty of a paper given its title and main contributions",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "the title (in English) of the academic paper your student will write.",
                },
                "contributions": {"type": "string",
                         "description": "a general description on the contributions of this paper in English."
                                                "If there are multiple contributions, index them with numbers."},
            },
            "required": ["title", "contributions"],
        },
    }
]

TOOLS = {"find_research_directions": find_research_directions, "find_references": find_references, "judge_novelty": judge_novelty}

class FindResearchDirectionsCheckInput(BaseModel):
    research_description: str = Field(..., description="a paragraph with details in English describing (1) what is the main problem you are trying to solve "
                                                       "(2) what is the main novelty of this idea (3) how to complete this research.")

class TitleDescriptionCheckInput(BaseModel):
    title: str = Field(..., description="the title of the academic paper your student will write in English.")
    contributions: str = Field(..., description="a general description on the contributions of this paper in English."
                                                "If there are multiple contributions, index them with numbers.")


class FindResearchDirectionsTool(BaseTool):
    name = "find_research_directions"
    description = """Useful when your student has already shown interests in a specific topic and provided a rough description of
    potential contributions and you need to help your student to dive this direction deeper for your student. 

                  """
    def _run(self, research_description: str):
        response = find_research_directions(research_description)
        return response

    def _arun(self, research_field: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = FindResearchDirectionsCheckInput


class JudgeNoveltyTool(BaseTool):
    name = "judge_novelty"
    description = """Useful when you need to evaluate the novelty of your student's idea.

                  """
    def _run(self, title: str, contributions: str):
        response = judge_novelty(title, contributions)
        return response

    def _arun(self, title: str, contributions: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = TitleDescriptionCheckInput

class FindReferencesTool(BaseTool):
    name = "find_references"
    description = """Useful when you need to find references for a paper.

                  """
    def _run(self, title: str, contributions: str):
        response = find_references(title, contributions)
        return response

    def _arun(self, title: str, contributions: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = TitleDescriptionCheckInput

