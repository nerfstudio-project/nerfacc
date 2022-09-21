#!/usr/bin/env python
"""Simple yaml debugger"""
import subprocess

import yaml
from rich.console import Console
from rich.style import Style

console = Console(width=120)

LOCAL_TESTS = [
    "Run license checks",
    "Run isort",
    "Run Black",
    "Python Pylint",
    "Test with pytest",
]


def run_command(command: str) -> bool:
    """Run a command kill actions if it fails

    Args:
        command: command to run
        continue_on_fail: whether to continue running commands if the current one fails.
    """
    ret_code = subprocess.call(command, shell=True)
    if ret_code != 0:
        console.print(f"[bold red]Error: `{command}` failed.")
    return ret_code == 0


def run_github_actions_file(filename: str):
    """Run a github actions file locally.

    Args:
        filename: Which yml github actions file to run.
    """
    with open(filename, "rb") as f:
        my_dict = yaml.safe_load(f)
    steps = my_dict["jobs"]["build"]["steps"]

    success = True

    for step in steps:
        if "name" in step and step["name"] in LOCAL_TESTS:
            compressed = step["run"].replace("\n", ";").replace("\\", "")
            compressed = compressed.replace("--check", "")
            curr_command = f"{compressed}"

            console.line()
            console.rule(f"[bold green]Running: {curr_command}")
            success = success and run_command(curr_command)
        else:
            skip_name = step["name"] if "name" in step else step["uses"]
            console.print(f"Skipping {skip_name}")

    # Code Testing
    console.line()
    console.rule("[bold green]Running pytest")
    success = success and run_command("pytest")

    # Add checks for building documentation
    console.line()
    console.rule("[bold green]Building Documentation")
    success = success and run_command(
        "cd docs/; make clean; make html SPHINXOPTS='-W;'"
    )

    if success:
        console.line()
        console.rule(characters="=")
        console.print(
            "[bold green]:TADA: :TADA: :TADA: ALL CHECKS PASSED :TADA: :TADA: :TADA:",
            justify="center",
        )
        console.rule(characters="=")
    else:
        console.line()
        console.rule(characters="=", style=Style(color="red"))
        console.print(
            "[bold red]:skull: :skull: :skull: ERRORS FOUND :skull: :skull: :skull:",
            justify="center",
        )
        console.rule(characters="=", style=Style(color="red"))


if __name__ == "__main__":
    run_github_actions_file(filename=".github/workflows/code_checks.yml")
