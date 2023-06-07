'''
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
'''
import os.path

from auto_backgrounds import generate_draft
import json, time
from utils.file_operations import make_archive


# GENERATOR_MAPPING = {"draft": generate_draft}
GENERATOR_MAPPING = {"draft": None}

def generator_wrapper(path_to_config_json):
    # Read configuration file and call corresponding function
    with open(path_to_config_json, "r", encoding='utf-8') as f:
        config = json.load(f)
    print("Configuration:", config)
    # generator = GENERATOR_MAPPING.get(config["generator"])
    generator = None
    if generator is None:
        # generate a fake ZIP file and upload
        time.sleep(150)
        zip_path = os.path.splitext(path_to_config_json)[0]+".zip"
        return make_archive(path_to_config_json, zip_path)

