import os

import typer
import yaml
from paddlenlp.utils.log import logger

logger.set_level("INFO")

text_classification_app = typer.Typer()
UIE_CONFIG_PATH = "./src/judge/text_classification/uie/configs"
UTC_CONFIG_PATH = "./src/judge/text_classification/utc/configs"


def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        logger.info(f"Loading the config file in {os.path.abspath(f.name)}...")
        data = yaml.load(f, Loader=yaml.Loader)
    return data


@text_classification_app.callback(invoke_without_command=True)
def classification_task(task_type: str):
    if task_type not in ["uie_dataprep", "uie_eval", "uie_train", "uie_infer", "utc_dataprep", "utc_eval", "utc_train", "utc_infer"]:
        raise ValueError(f"Cannot find the task: {task_type}")

    exec("from ." + task_type.replace("_", ".") + " import main")
    config_path = UIE_CONFIG_PATH if "uie" in task_type else UTC_CONFIG_PATH
    args = load_config(yaml_file=os.path.join(config_path, task_type.split("_")[1] + ".yaml"))
    eval("main")(args=args)
