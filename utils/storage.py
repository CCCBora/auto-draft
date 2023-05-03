# This script `storage.py` is used to handle the cloud storage.
#   `upload_file`:
#   `list_all_files`:
#   `download_file`:

import os
import boto3

access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
bucket_name = "hf-storage"

if (access_key_id is not None) and (secret_access_key is not None):
    session = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )

    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket_name)


    def upload_file(file_name, target_name=None):
        if target_name is None:
            target_name = file_name
        try:
            s3.meta.client.upload_file(Filename=file_name, Bucket=bucket_name, Key=target_name)
            print(f"The file {file_name} has been uploaded!")
        except:
            print("Uploading failed!")

    def list_all_files():
        return [obj.key for obj in bucket.objects.all()]

    def download_file(file_name):
        ''' Download `file_name` from the bucket.
        Bucket (str) – The name of the bucket to download from.
        Key (str) – The name of the key to download from.
        Filename (str) – The path to the file to download to.
        '''
        try:
            s3.meta.client.download_file(Bucket=bucket_name, Key=file_name, Filename=file_name)
            print(f"The file {file_name} has been downloaded!")
        except:
            print("Uploading failed!")