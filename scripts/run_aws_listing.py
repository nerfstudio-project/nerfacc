import argparse
import os

from boto3 import client

parser = argparse.ArgumentParser()
parser.add_argument("--access_key_id", type=str, required=True)
parser.add_argument("--secret_access_key", type=str, required=True)
parser.add_argument("--bucket", type=str, required=True)
parser.add_argument("--region", type=str, required=True)
args = parser.parse_args()

URL = f"https://{args.bucket}.s3.{args.region}.amazonaws.com/"

s3 = client(
    "s3",
    aws_access_key_id=args.access_key_id,
    aws_secret_access_key=args.secret_access_key,
)

responses = s3.list_objects_v2(Bucket=args.bucket, Prefix="whl/")["Contents"]

subdirectories = {}
for data in responses:
    splits = data["Key"].split("/")
    if len(splits) == 3:
        subdirectories[splits[1]] = []

for dir in subdirectories.keys():
    responses = s3.list_objects_v2(Bucket=args.bucket, Prefix=f"whl/{dir}")[
        "Contents"
    ]
    for data in responses:
        splits = data["Key"].split("/")
        if len(splits) == 3:
            subdirectories[dir].append(splits[2])

for dir, files in subdirectories.items():
    lines = ""
    for file in files:
        href = os.path.join(URL, "whl", dir, file)
        lines += f"<a href='{href}'>{file}</a>\n<br>\n"

    html = f"<html>\n<head></head>\n<body>\n{lines}\n</body>\n</html>\n"
    html_file = f"/tmp/{dir}.html"
    with open(html_file, "w") as f:
        f.write(html)

    s3.upload_file(
        html_file,
        args.bucket,
        f"whl/{dir}.html",
        ExtraArgs={"ContentType": "text/html"},
    )
