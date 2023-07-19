"""
This script is used to wrap all generation methods together.

todo:
    A worker keeps running on the server. Monitor the Amazon SQS. Once receive a new message, do the following:
        Download the corresponding configuration files on S3.
        Change Task status from Pending to Running.
        Call `generator_wrapper` and wait for the outputs.
        If `generator_wrapper` returns results:
            evaluate the results; compile it; upload results to S3 ... Change Task status from Running to Completed.
            If anything goes wrong, raise Error.
        If `generator_wrapper` returns nothing or Timeout, or raise any error:
            Change Task status from Running to Failed.
"""
from auto_generators import generate_draft
from utils.file_operations import make_archive
import yaml
import uuid


def remove_special_characters(s):
    return ''.join(c for c in s if c.isalnum() or c.isspace() or c == ',')


def generator_wrapper(config):
    if not isinstance(config, dict):
        with open(config, "r") as file:
            config = yaml.safe_load(file)
    title = config["paper"]["title"]
    generator = config["generator"]
    if generator == "auto_draft":
        folder = generate_draft(title, config["paper"]["description"],
                                tldr=config["references"]["tldr"],
                                max_kw_refs=config["references"]["max_kw_refs"],
                                refs=config["references"]["refs"],
                                max_tokens_ref=config["references"]["max_tokens_ref"],
                                knowledge_database=config["domain_knowledge"]["knowledge_database"],
                                max_tokens_kd=config["domain_knowledge"]["max_tokens_kd"],
                                query_counts=config["domain_knowledge"]["query_counts"],
                                sections=config["output"]["selected_sections"],
                                model=config["output"]["model"],
                                template=config["output"]["template"],
                                prompts_mode=config["output"]["prompts_mode"],
                                )
    else:
        raise NotImplementedError(f"The generator {generator} has not been supported yet.")
    # todo: post processing: algorithms (in methodology), translate to Chinese, compile PDF ...
    filename = remove_special_characters(title).replace(" ", "_") + uuid.uuid1().hex + ".zip"
    return make_archive(folder, filename)


if __name__ == "__main__":
    pass
    # with open("configurations/default.yaml", 'r') as file:
    #     config = yaml.safe_load(file)
    # print(config)
    # generator_wrapper(config)
