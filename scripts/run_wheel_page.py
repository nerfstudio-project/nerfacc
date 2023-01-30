import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--pt", type=str, required=True)
parser.add_argument("--cu", type=str, required=True)
args = parser.parse_args()

folder = f"torch-{args.pt}+{args.cu}"
files = [f for f in os.listdir(folder) if f.endswith(".whl")]

lines = ""
for file in files:
    href = os.path.join(folder, file)
    href = href.replace("+", "%2B")
    lines += f"<a href='{href}'>{file}</a>\n<br>\n"

html = f"<html>\n<head></head>\n<body>\n{lines}\n</body>\n</html>\n"
with open("index.html", "w") as f:
    f.write(html)
