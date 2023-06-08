'''
This script is used to generate the most relevant papers of a given title.
    - Search for as many as possible references. For 10~15 keywords, 10 references each.
    - Sort the results from most relevant to least relevant.
    - Return the most relevant using token size.

Note: we do not use this function in auto-draft function. It has been integrated in that.
'''

import os.path
import json
from utils.references import References
from section_generator import keywords_generation # section_generation_bg,  #, figures_generation, section_generation
import itertools
from gradio_client import Client


def generate_raw_references(title, description="",
                            bib_refs=None, tldr=False, max_kw_refs=10,
                            save_to="ref.bib"):
    # load pre-provided references
    ref = References(title, bib_refs)

    # generate multiple keywords for searching
    input_dict = {"title": title, "description": description}
    keywords, usage = keywords_generation(input_dict)
    keywords = list(keywords)
    comb_keywords = list(itertools.combinations(keywords, 2))
    for comb_keyword in comb_keywords:
        keywords.append(" ".join(comb_keyword))
    keywords = {keyword:max_kw_refs for keyword in keywords}
    print(f"keywords: {keywords}\n\n")

    ref.collect_papers(keywords, tldr=tldr)
    paper_json = ref.to_json()

    with open(save_to, "w") as f:
        json.dump(paper_json, f)

    return save_to, ref # paper_json

def generate_top_k_references(title, description="",
                            bib_refs=None, tldr=False, max_kw_refs=10,  save_to="ref.bib", top_k=5):
    json_path, ref_raw = generate_raw_references(title, description, bib_refs, tldr, max_kw_refs,  save_to)
    json_content = ref_raw.to_json()

    client = Client("https://shaocongma-evaluate-specter-embeddings.hf.space/")
    result = client.predict(
        title,  # str  in 'Title' Textbox component
        json_path,  # str (filepath or URL to file) in 'Papers JSON (as string)' File component
        top_k,  # int | float (numeric value between 1 and 50) in 'Top-k Relevant Papers' Slider component
        api_name="/get_k_relevant_papers"
    )
    with open(result) as f:
        result = json.load(f)
    return result


if __name__ == "__main__":
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")

    title = "Using interpretable boosting algorithms for modeling environmental and agricultural data"
    description = ""
    save_to = "paper.json"
    save_to, paper_json = generate_raw_references(title, description, save_to=save_to)

    print("`paper.json` has been generated. Now evaluating its similarity...")

    k = 5
    client = Client("https://shaocongma-evaluate-specter-embeddings.hf.space/")
    result = client.predict(
        title,  # str  in 'Title' Textbox component
        save_to,  # str (filepath or URL to file) in 'Papers JSON (as string)' File component
        k,  # int | float (numeric value between 1 and 50) in 'Top-k Relevant Papers' Slider component
        api_name="/get_k_relevant_papers"
    )

    with open(result) as f:
        result = json.load(f)

    print(result)

    save_to = "paper2.json"
    with open(save_to, "w") as f:
        json.dump(result, f)