import argparse
from collections import defaultdict

from boto3 import resource

parser = argparse.ArgumentParser()
parser.add_argument("--bucket", type=str, required=True)
parser.add_argument("--region", type=str, required=True)
args = parser.parse_args()

ROOT_URL = f"https://{args.bucket}.s3.{args.region}.amazonaws.com/whl"
html = "<!DOCTYPE html>\n<html>\n<body>\n{}\n</body>\n</html>"
href = '  <a href="{}">{}</a><br/>'
html_args = {
    "ContentType": "text/html",
    "CacheControl": "max-age=300",
    "ACL": "public-read",
}

bucket = resource("s3").Bucket(name="nerfacc-bucket")

wheels_dict = defaultdict(list)
for obj in bucket.objects.filter(Prefix="whl"):
    if obj.key[-3:] != "whl":
        continue
    torch_version, wheel = obj.key.split("/")[-2:]
    wheel = f"{torch_version}/{wheel}"
    wheels_dict[torch_version].append(wheel)

index_html = html.format(
    "\n".join(
        [
            href.format(
                f"{torch_version}.html".replace("+", "%2B"), torch_version
            )
            for torch_version in wheels_dict
        ]
    )
)

with open("index.html", "w") as f:
    f.write(index_html)
bucket.Object("whl/index.html").upload_file("index.html", html_args)

for torch_version, wheel_names in wheels_dict.items():
    torch_version_html = html.format(
        "\n".join(
            [
                href.format(
                    f"{ROOT_URL}/{wheel_name}".replace("+", "%2B"),
                    wheel_name.split("/")[-1],
                )
                for wheel_name in wheel_names
            ]
        )
    )

    with open(f"{torch_version}.html", "w") as f:
        f.write(torch_version_html)
    bucket.Object(f"whl/{torch_version}.html").upload_file(
        f"{torch_version}.html", html_args
    )
