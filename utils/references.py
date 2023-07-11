# Each `paper` is a dictionary containing:
#       (1) paper_id (2) title (3) authors (4) year (5) link (6) abstract (7) journal (8) embeddings
#
# Generate references:
#   `Reference` class:
#       1. Two methods to load papers:
#           1.1. Read a given string including paper titles separated by `,`
#           1.2. Read a .bib file
#       2. Given some keywords; use Semantic Scholar API to find papers.
#       3. Generate bibtex from the selected papers. --> to_bibtex()
#       4. Generate prompts from the selected papers: --> to_prompts()
#               A sample prompt: {"paper_id": "paper summary"}
#       5. Generate json from the selected papers. --> to_json()

import itertools
import json
import re
import uuid
from typing import Dict, List, Optional, Union

import arxiv
import bibtexparser
import numpy as np
import requests
import tiktoken
from numpy.linalg import norm
from scholarly import ProxyGenerator
from scholarly import scholarly

# used to evaluate embeddings
URL = "https://model-apis.semanticscholar.org/specter/v1/invoke"
MAX_BATCH_SIZE = 16
MAX_ATTEMPTS = 20

# `tokenizer`: used to count how many tokens
tokenizer_name = tiktoken.encoding_for_model('gpt-4')
tokenizer = tiktoken.get_encoding(tokenizer_name.name)


######################################################################################################################
# Some basic tools
######################################################################################################################
def remove_special_characters(s):
    return ''.join(c for c in s if c.isalnum() or c.isspace() or c == ',')


def remove_newlines(serie):
    # This function is applied to the abstract of each paper to reduce the length of prompts.
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    serie = serie.replace('  ', ' ')
    return serie


def evaluate_cosine_similarity(v1, v2):
    try:
        return np.dot(v1, v2) / (norm(v1) * norm(v2))
    except ValueError:
        return 0.0


def chunks(lst, chunk_size=MAX_BATCH_SIZE):
    """Splits a longer list to respect batch size"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i: i + chunk_size]


def embed(papers):
    embeddings_by_paper_id: Dict[str, List[float]] = {}
    for chunk in chunks(papers):
        # Allow Python requests to convert the data above to JSON
        response = requests.post(URL, json=chunk)

        if response.status_code != 200:
            raise RuntimeError("Sorry, something went wrong, please try later!")

        for paper in response.json()["preds"]:
            embeddings_by_paper_id[paper["paper_id"]] = paper["embedding"]

    return embeddings_by_paper_id


def get_embeddings(paper_title, paper_description):
    output = [{"title": paper_title, "abstract": paper_description, "paper_id": "target_paper"}]
    emb_vector = embed(output)["target_paper"]
    target_paper = output[0]
    target_paper["embeddings"] = emb_vector
    return target_paper


def get_embeddings_vector(paper_title, paper_description):
    output = [{"title": paper_title, "abstract": paper_description, "paper_id": "target_paper"}]
    emb_vector = embed(output)["target_paper"]
    return emb_vector


def get_top_k(papers_dict, paper_title, paper_description, k=None):
    # returns the top k papers most similar to the target paper
    target_paper = get_embeddings(paper_title, paper_description)
    papers = papers_dict  # must include embeddings

    # if k < len(papers_json), return k most relevant papers
    # if k >= len(papers_json) or k is None, return all papers
    max_num_papers = len(papers)
    if k is None:
        k = max_num_papers
    num_papers = min(k, max_num_papers)

    # evaluate the cosine similarity for each paper
    target_embedding_vector = target_paper["embeddings"]

    for k in papers:
        v = papers[k]
        embedding_vector = v["embeddings"]
        cos_sim = evaluate_cosine_similarity(embedding_vector, target_embedding_vector)
        papers[k]["cos_sim"] = cos_sim

    # return the best k papers
    sorted_papers = {k: v for k, v in sorted(papers.items(), key=lambda x: x[1]["cos_sim"], reverse=True)[:num_papers]}
    for key in sorted_papers:
        sorted_papers[key].pop("embeddings", None)
    return sorted_papers


def search_paper_abstract(title):
    pg = ProxyGenerator()
    success = pg.FreeProxies()  # pg.ScraperAPI("921b16f94d701308b9d9b4456ddde155")
    if success:
        try:
            scholarly.use_proxy(pg)
            # input the title of a paper, return its abstract
            search_query = scholarly.search_pubs(title)
            found_paper = next(search_query)
        except:
            return ""
    else:
        return ""
        # raise RuntimeError("ScraperAPI fails.")
    return remove_newlines(found_paper['bib']['abstract'])


def tiktoken_len(text):
    # evaluate how many tokens for the given text
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)


######################################################################################################################
# Academic search tools
######################################################################################################################
def externalIds2link(externalIds):
    # Sample externalIds:
    #   "{'MAG': '2932819148', 'DBLP': 'conf/icml/HaarnojaZAL18', 'ArXiv': '1801.01290', 'CorpusId': 28202810}"
    if externalIds:
        # Supports ArXiv, MAG, ACL, PubMed, Medline, PubMedCentral, DBLP, DOI
        # priority: DBLP > arXiv > (todo: MAG > CorpusId > DOI > ACL > PubMed > Mdeline > PubMedCentral)
        # DBLP
        dblp_id = externalIds.get('DBLP')
        if dblp_id is not None:
            dblp_link = f"dblp.org/rec/{dblp_id}"
            return dblp_link
        # arXiv
        arxiv_id = externalIds.get('ArXiv')
        if arxiv_id is not None:
            arxiv_link = f"arxiv.org/abs/{arxiv_id}"
            return arxiv_link
        return ""
    else:
        # if this is an empty dictionary, return an empty string
        return ""


def search_paper_arxiv(title):
    search = arxiv.Search(
        query=title,
        max_results=1,
        sort_by=arxiv.SortCriterion.Relevance
    )
    try:
        #       (1) paper_id (2) title (3) authors (4) year (5) link (6) abstract (7) journal (8) embeddings
        result = next(search.results())
        title = result.title
        authors = " and ".join([author.name for author in result.authors])
        year = str(result.updated.now().year)
        link = result.pdf_url
        abstract = result.summary
        journal = f"Arxiv: {result.entry_id}"
        paper_id = result.authors[0].name.replace(" ", "")[:4] + year + title[:6].replace(" ", "")
        paper_id = paper_id.lower()

        paper = {"paper_id": paper_id,
                 "title": title,
                 "authors": authors,
                 "year": year,
                 "link": link,
                 "abstract": abstract,
                 "journal": journal}
    except StopIteration:
        paper = {}
    return paper


def search_paper_ss(title):
    fields = ["title", "abstract", "venue", "year", "authors", "tldr", "externalIds"]
    limit = 1
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit={limit}&fields={",".join(fields)}'
    # headers = {"Accept": "*/*", "x-api-key": constants.S2_KEY}
    headers = {"Accept": "*/*"}
    response = requests.get(url, headers=headers, timeout=30)
    results = response.json()
    if results['total'] == 0:
        return {}
    raw_paper = results['data'][0]
    if raw_paper['tldr'] is not None:
        abstract = raw_paper['tldr']['text']
    elif raw_paper['abstract'] is not None:
        abstract = remove_newlines(raw_paper['abstract'])
    else:
        abstract = ""

    authors = [author['name'] for author in raw_paper['authors']]
    authors_str = " and ".join(authors)
    year_str = str(raw_paper['year'])
    title = raw_paper['title']

    paper_id = authors_str.replace(" ", "")[:4] + year_str + title[:6].replace(" ", "")

    # some journal may contain &; replace it. e.g. journal={IEEE Power & Energy Society General Meeting}
    journal = remove_special_characters(raw_paper['venue'])
    if not journal:
        journal = "arXiv preprint"
    link = externalIds2link(raw_paper['externalIds'])
    paper = {
        "paper_id": paper_id,
        "title": title,
        "abstract": abstract,
        "link": link,
        "authors": authors_str,
        "year": year_str,
        "journal": journal
    }
    return paper


def search_paper_scrape(title):
    pg = ProxyGenerator()
    success = pg.ScraperAPI("921b16f94d701308b9d9b4456ddde155")
    if success:
        try:
            scholarly.use_proxy(pg)
            # input the title of a paper, return its abstract
            search_query = scholarly.search_pubs(title)
            found_paper = next(search_query)
            url = found_paper['pub_url']

            result = found_paper['bib']

            title = result['title']
            authors = " and ".join(result['author'])
            year = str(result['pub_year'])
            journal = result['pub_year']
            abstract = result['abstract']

            paper_id = authors.replace(" ", "")[:4] + year + title[:6].replace(" ", "")
            paper = {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "link": url,
                "authors": authors,
                "year": year,
                "journal": journal
            }
            return paper
        except StopIteration:
            return {}


def search_paper(title, verbose=True):
    if verbose:
        print(f"Searching {title}...")
    # try Semantic Scholar first
    paper = search_paper_ss(title)
    if not paper:
        paper = search_paper_arxiv(title)
    if not paper:
        paper = search_paper_scrape(title)
    if paper:
        paper["embeddings"] = get_embeddings_vector(paper_title=paper['title'], paper_description=paper['abstract'])
    if verbose:
        print(f"Search result: {paper}.")
    return paper


def load_papers_from_bibtex(bib_file_path):
    with open(bib_file_path) as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)
    if len(bib_database.entries) == 0:
        return []
    else:
        bib_papers = []
        for bibitem in bib_database.entries:
            # Add each paper to `bib_papers`
            paper_id = bibitem.get("ID")
            title = bibitem.get("title")
            if title is None:
                continue
            journal = bibitem.get("journal")
            year = bibitem.get("year")
            author = bibitem.get("author")
            abstract = bibitem.get("abstract")
            if abstract is None:
                abstract = search_paper_abstract(title)
            result = {
                "paper_id": paper_id,
                "title": title,
                "link": "",
                "abstract": abstract,
                "authors": author,
                "year": year,
                "journal": journal
            }
            bib_papers.append(result)
        return bib_papers


def load_papers_from_text(text):
    # split text by comma
    titles = [part.strip() for part in text.split(',')]
    titles = [remove_special_characters(title) for title in titles]
    papers = []
    if len(titles) > 0:
        for title in titles:
            paper = search_paper(title)
            if paper:
                papers.append(paper)
        return papers
    else:
        return []


######################################################################################################################
# Semantic Scholar (SS) API
######################################################################################################################
def ss_search(keywords, limit=20, fields=None):
    # space between the  query to be removed and replaced with +
    if fields is None:
        fields = ["title", "abstract", "venue", "year", "authors", "tldr", "embedding", "externalIds"]
    keywords = keywords.lower()
    keywords = keywords.replace(" ", "+")
    url = f'https://api.semanticscholar.org/graph/v1/paper/search?query={keywords}&limit={limit}&fields={",".join(fields)} '
    # headers = {"Accept": "*/*", "x-api-key": constants.S2_KEY}
    headers = {"Accept": "*/*"}

    response = requests.get(url, headers=headers, timeout=30)
    return response.json()


def _collect_papers_ss(keyword, counts=3, tldr=False):
    def extract_paper_id(last_name, year_str, title):
        pattern = r'^\w+'
        words = re.findall(pattern, title)
        # return last_name + year_str + title.split(' ', 1)[0]
        try:
            output = last_name + year_str + words[0]
        except IndexError:
            output = last_name + year_str + title[:4]
        return output

    def extract_author_info(raw_authors):
        authors = [author['name'] for author in raw_authors]

        authors_str = " and ".join(authors)
        try:
            last_name = authors[0].split()[-1]
            last_name = last_name.replace("'", "")
        except IndexError:
            last_name = "ma"
        # pattern = r'^\w+'
        # last_name = re.findall(pattern, authors[0])
        return authors_str, last_name

    def parse_search_results(search_results_ss):
        if len(search_results_ss) == 0:
            return []

        # turn the search result to a list of paper dictionary.
        papers_ss = []
        for raw_paper in search_results_ss:
            if raw_paper["abstract"] is None:
                continue

            authors_str, last_name = extract_author_info(raw_paper['authors'])
            year_str = str(raw_paper['year'])
            title = raw_paper['title']

            # some journal may contain &; replace it. e.g. journal={IEEE Power & Energy Society General Meeting}
            journal = raw_paper['venue'].replace("&", "\\&")
            if not journal:
                journal = "arXiv preprint"

            paper_id = extract_paper_id(last_name, year_str, title).lower()
            link = externalIds2link(raw_paper['externalIds'])

            if tldr and raw_paper['tldr'] is not None:
                abstract = raw_paper['tldr']['text']
            else:
                abstract = remove_newlines(raw_paper['abstract'])

            # some papers have no embeddings; handle this case
            embeddings_dict = raw_paper.get('embedding')
            if embeddings_dict is None:
                continue
            else:
                embeddings = raw_paper['embedding']['vector']
            result = {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "link": link,
                "authors": authors_str,
                "year": year_str,
                "journal": journal,
                "embeddings": embeddings
            }
            papers_ss.append(result)
        return papers_ss

    raw_results = ss_search(keyword, limit=counts)
    if raw_results is not None:
        search_results = raw_results.get("data")
        if search_results is None:
            search_results = []
    else:
        search_results = []
    results = parse_search_results(search_results)
    return results


######################################################################################################################
# References Class
######################################################################################################################

class References:
    def __init__(self,
                 title: str,
                 load_papers: Optional[str] = None,
                 load_bibtex: Optional[str] = None,
                 description: str = ""
                 ):
        self.papers = {}
        if load_bibtex is not None:
            self.papers["load_from_bibtex"] = load_papers_from_bibtex(load_bibtex)
        if load_papers is not None:
            self.papers["load_from_text"] = load_papers_from_text(load_papers)

        self.title = title
        self.description = description

    def generate_keywords_dict(self) -> Dict[str, int]:
        keywords_dict = {}
        for k in self.papers:
            keywords_dict[k] = len(self.papers[k])
        return keywords_dict

    def collect_papers(self, keywords_dict: Dict[str, int], tldr: bool = False) -> None:
        """
        Collect as many papers as possible

        keywords_dict:
            {"machine learning": 5, "language model": 2};
            the first is the keyword, the second is how many references are needed.
        """
        keywords = list(keywords_dict)
        comb_keywords = list(itertools.combinations(keywords, 2))
        for comb_keyword in comb_keywords:
            keywords.append(" ".join(comb_keyword))
        for key in keywords:
            self.papers[key] = _collect_papers_ss(key, 10, tldr)

    def to_bibtex(self, path_to_bibtex: str = "ref.bib") -> List[str]:
        """
        Turn the saved paper list into bibtex file "ref.bib". Return a list of all `paper_id`.
        """
        papers = self._get_papers(keyword="_all")

        num_papers = len(papers)
        print(f"{num_papers} papers will be added to `ref.bib`.")
        # clear the bibtex file
        with open(path_to_bibtex, "w", encoding="utf-8") as file:
            file.write("")

        bibtex_entries = []
        paper_ids = []
        seen = set()
        for paper in papers:
            if paper["paper_id"] in seen:
                continue
            else:
                seen.add(paper["paper_id"])
            bibtex_entry = f"""@article{{{paper["paper_id"]},
          title = {{{paper["title"]}}},
          author = {{{paper["authors"]}}}, 
          journal={{{paper["journal"]}}}, 
          year = {{{paper["year"]}}}, 
          url = {{{paper["link"]}}}
        }}"""
            bibtex_entries.append(bibtex_entry)
            paper_ids.append(paper["paper_id"])
            # Save the generated BibTeX entries to a file
            with open(path_to_bibtex, "a", encoding="utf-8") as file:
                file.write(bibtex_entry)
                file.write("\n\n")
                # print(f'{paper["paper_id"]} has been added to `ref.bib`.')
        return paper_ids

    def _get_papers(self, keyword="_all"):
        if keyword == "_all":
            papers = []
            for k, v in self.papers.items():
                papers = papers + v
        else:
            papers = self.papers["keyword"]
        return papers

    def to_prompts(self, keyword: str = "_all", max_tokens: int = 2048):
        # `prompts`:
        #   {"paper1_bibtex_id": "paper_1_abstract", "paper2_bibtex_id": "paper2_abstract"}
        #   this will be used to instruct GPT model to cite the correct bibtex entry.

        # two steps:
        #   1. Sort everything from most relevant to less relevant
        #   2. Add paper to prompts until max_tokens
        json_path = str(uuid.uuid1()) + ".json"
        papers_json = self.to_json()
        with open(json_path, "w") as f:
            json.dump(papers_json, f)
        try:
            # Use external API to obtain the most relevant papers
            title = self.title
            description = self.description
            result = get_top_k(papers_json, title, description)
            result = [item for key, item in result.items()]
        except Exception as e:
            print(f"Error occurs during calling external API: {e}\n")
            print("Use default method instead!")
            result = self._get_papers(keyword)
        prompts = {}
        tokens = 0
        for paper in result:
            abstract = paper.get("abstract")
            if abstract is not None and isinstance(abstract, str):
                prompts[paper["paper_id"]] = paper["abstract"]
                tokens += tiktoken_len(paper["abstract"])
            else:
                prompts[paper["paper_id"]] = " "
            if tokens >= max_tokens:
                break
        return prompts

    def to_json(self, keyword: str = "_all"):
        papers = self._get_papers(keyword)
        papers_json = {}
        for paper in papers:
            papers_json[paper["paper_id"]] = paper
        return papers_json
