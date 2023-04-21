# Generate references
#   1. select most correlated references from "references" dataset or Arxiv search engine.
#   2. Generate bibtex from the selected papers. --> to_bibtex()
#   3. Generate prompts from the selected papers: --> to_prompts()
#       {"paper_id": "paper summary"}


import requests
import re

def _collect_papers_arxiv(keyword, counts=3):
    #
    # The following codes are used to generate the most related papers
    #
    # Build the arXiv API query URL with the given keyword and other parameters
    def build_query_url(keyword, results_limit=3, sort_by="relevance", sort_order="descending"):
        base_url = "http://export.arxiv.org/api/query?"
        query = f"search_query=all:{keyword}&start=0&max_results={results_limit}"
        query += f"&sortBy={sort_by}&sortOrder={sort_order}"
        return base_url + query

    # Fetch search results from the arXiv API using the constructed URL
    def fetch_search_results(query_url):
        response = requests.get(query_url)
        return response.text

    # Parse the XML content of the API response to extract paper information
    def parse_results(content):
        from xml.etree import ElementTree as ET

        root = ET.fromstring(content)
        namespace = "{http://www.w3.org/2005/Atom}"
        entries = root.findall(f"{namespace}entry")

        results = []
        for entry in entries:
            title = entry.find(f"{namespace}title").text
            link = entry.find(f"{namespace}id").text
            summary = entry.find(f"{namespace}summary").text

            # Extract the authors
            authors = entry.findall(f"{namespace}author")
            author_list = []
            for author in authors:
                name = author.find(f"{namespace}name").text
                author_list.append(name)
            authors_str = " and ".join(author_list)

            # Extract the year
            published = entry.find(f"{namespace}published").text
            year = published.split("-")[0]

            founds = re.search(r'\d+\.\d+', link)
            if founds is None:
                # some links are not standard; such as "https://arxiv.org/abs/cs/0603127v1".
                # will be solved in the future.
                continue
            else:
                arxiv_id = founds.group(0)
            journal = f"arXiv preprint arXiv:{arxiv_id}"
            result = {
                "paper_id": arxiv_id,
                "title": title,
                "link": link,
                "abstract": summary,
                "authors": authors_str,
                "year": year,
                "journal": journal
            }
            results.append(result)

        return results

    query_url = build_query_url(keyword, counts)
    content = fetch_search_results(query_url)
    results = parse_results(content)
    return results

# Each `paper` is a dictionary containing (1) paper_id (2) title (3) authors (4) year (5) link (6) abstract (7) journal
class References:
    def __init__(self, load_papers = ""):
        if load_papers:
            # todo: read a json file from the given path
            #       this could be used to support pre-defined references
            pass
        else:
            self.papers = []

    def collect_papers(self, keywords_dict, method="arxiv"):
        """
        keywords_dict:
            {"machine learning": 5, "language model": 2};
            the first is the keyword, the second is how many references are needed.
        """
        match method:
            case "arxiv":
                process =_collect_papers_arxiv
            case _:
                raise NotImplementedError("Other sources have not been not supported yet.")
        for key, counts in keywords_dict.items():
            self.papers = self.papers + process(key, counts)

        seen = set()
        papers = []
        for paper in self.papers:
            paper_id = paper["paper_id"]
            if paper_id not in seen:
                seen.add(paper_id)
                papers.append(paper)
        self.papers = papers

    def to_bibtex(self, path_to_bibtex="ref.bib"):
        """
        Turn the saved paper list into bibtex file "ref.bib". Return a list of all `paper_id`.
        """
        papers = self.papers

        # clear the bibtex file
        with open(path_to_bibtex, "w", encoding="utf-8") as file:
            file.write("")

        bibtex_entries = []
        paper_ids = []
        for paper in papers:
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
        return paper_ids

    def to_prompts(self):
        # `prompts`:
        #   {"paper1_bibtex_id": "paper_1_abstract", "paper2_bibtex_id": "paper2_abstract"}
        #   this will be used to instruct GPT model to cite the correct bibtex entry.
        prompts = {}
        for paper in self.papers:
            prompts[paper["paper_id"]] = paper["abstract"]
        return prompts

if __name__ == "__main__":
    refs = References()
    keywords_dict = {
  "Deep Q-Networks": 5,
  "Policy Gradient Methods": 4,
  "Actor-Critic Algorithms": 4,
  "Model-Based Reinforcement Learning": 3,
  "Exploration-Exploitation Trade-off": 2
}
    refs.collect_papers(keywords_dict)
    for p in refs.papers:
        print(p["paper_id"])