'''
This script is only used for service-side host.
'''
import boto3
import os, time
from wrapper import generator_wrapper
from sqlalchemy import create_engine, Table, MetaData, update, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect

QUEUE_URL = os.getenv('QUEUE_URL')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')
DB_STRING = os.getenv('DATABASE_STRING')

# Create engine
ENGINE = create_engine(DB_STRING)
SESSION = sessionmaker(bind=ENGINE)


#######################################################################################################################
# Amazon SQS Handler
#######################################################################################################################
def get_sqs_client():
    sqs = boto3.client('sqs', region_name="us-east-2",
                       aws_access_key_id=AWS_ACCESS_KEY_ID,
                       aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    return sqs


def receive_message():
    sqs = get_sqs_client()
    message = sqs.receive_message(QueueUrl=QUEUE_URL)
    if message.get('Messages') is not None:
        receipt_handle = message['Messages'][0]['ReceiptHandle']
    else:
        receipt_handle = None
    return message, receipt_handle


def delete_message(receipt_handle):
    sqs = get_sqs_client()
    response = sqs.delete_message(QueueUrl=QUEUE_URL, ReceiptHandle=receipt_handle)
    return response


#######################################################################################################################
# AWS S3 Handler
#######################################################################################################################
def get_s3_client():
    access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    session = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )
    s3 = session.resource('s3')
    bucket = s3.Bucket(BUCKET_NAME)
    return s3, bucket


def upload_file(file_name, target_name=None):
    s3, _ = get_s3_client()

    if target_name is None:
        target_name = file_name
    s3.meta.client.upload_file(Filename=file_name, Bucket=BUCKET_NAME, Key=target_name)
    print(f"The file {file_name} has been uploaded!")


def download_file(file_name):
    """ Download `file_name` from the bucket.
    Bucket (str) – The name of the bucket to download from.
    Key (str) – The name of the key to download from.
    Filename (str) – The path to the file to download to.
    """
    s3, _ = get_s3_client()
    s3.meta.client.download_file(Bucket=BUCKET_NAME, Key=file_name, Filename=os.path.basename(file_name))
    print(f"The file {file_name} has been downloaded!")


#######################################################################################################################
# AWS SQL Handler
#######################################################################################################################
def modify_status(task_id, new_status):
    session = SESSION()
    metadata = MetaData()
    task_to_update = task_id
    task_table = Table('task', metadata, autoload_with=ENGINE)
    stmt = select(task_table).where(task_table.c.task_id == task_to_update)
    # Execute the statement
    with ENGINE.connect() as connection:
        result = connection.execute(stmt)

        # Fetch the first result (if exists)
        task_data = result.fetchone()

        # If user_data is not None, the user exists and we can update the password
        if task_data:
            # Update statement
            stmt = (
                update(task_table).
                    where(task_table.c.task_id == task_to_update).
                    values(status=new_status)
            )
            # Execute the statement and commit
            result = connection.execute(stmt)
            connection.commit()
    # Close the session
    session.close()

#######################################################################################################################
# Pipline
#######################################################################################################################
def pipeline(message_count=0, query_interval=10):
    # status: 0 - pending (default), 1 - running, 2 - completed, 3 - failed

    # Query a message from SQS
    msg, handle = receive_message()
    if handle is None:
        print("No message in SQS. ")
        time.sleep(query_interval)
    else:
        print("===============================================================================================")
        print(f"MESSAGE COUNT: {message_count}")
        print("===============================================================================================")
        config_s3_path = msg['Messages'][0]['Body']
        config_s3_dir = os.path.dirname(config_s3_path)
        config_local_path = os.path.basename(config_s3_path)
        task_id, _ = os.path.splitext(config_local_path)

        print("Initializing ...")
        print("Configuration file on S3: ", config_s3_path)
        print("Configuration file on S3 (Directory): ", config_s3_dir)
        print("Local file path: ", config_local_path)
        print("Task id: ", task_id)

        print(f"Success in receiving message: {msg}")
        print(f"Configuration file path: {config_s3_path}")

        # Process the downloaded configuration file
        download_file(config_s3_path)
        modify_status(task_id, 1)  # status: 0 - pending (default), 1 - running, 2 - completed, 3 - failed
        delete_message(handle)
        print(f"Success in the initialization. Message deleted.")

        print("Running ...")
        # try:
        zip_path = generator_wrapper(config_local_path)
        # Upload the generated file to S3
        upload_to = os.path.join(config_s3_dir, zip_path).replace("\\", "/")

        print("Local file path (ZIP): ", zip_path)
        print("Upload to S3: ", upload_to)
        upload_file(zip_path, upload_to)
        modify_status(task_id, 2) # status: 0 - pending (default), 1 - running, 2 - completed, 3 - failed, 4 - deleted
        print(f"Success in generating the paper.")

        # Complete.
        print("Task completed.")


def initialize_everything():
    # Clear S3

    # Clear SQS
    pass


if __name__ == "__main__":
    pipeline()
