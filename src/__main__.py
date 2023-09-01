"""
Entrypoint module, in case you use `python -m judge`.


Why does this file exist, and why __main__? For more info, read:

- https://www.python.org/dev/peps/pep-0338/
- https://docs.python.org/2/using/cmdline.html#cmdoption-m
- https://docs.python.org/3/using/cmdline.html#cmdoption-m
"""

import typer

from judge.information_extraction import information_extraction_app, load_config
from judge.text_classification import text_classification_app
from src.judge.merge import do_merge

task = typer.Typer()
task.add_typer(information_extraction_app, name="information_extraction")
task.add_typer(text_classification_app, name="text_classification")


@task.command(name="merge")
def merge_command():
    args = load_config("./src/judge/merge_config.yaml")
    do_merge(args=args)


if __name__ == "__main__":
    task()
