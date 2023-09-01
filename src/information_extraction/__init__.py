import os

import typer
import yaml
from paddlenlp.utils.log import logger

logger.set_level("INFO")

information_extraction_app = typer.Typer()

BASE_CONFIG_PATH = "./src/judge/information_extraction/configs/"
# ENTITY_TYPE = ["精神慰撫金額", "醫療費用", "薪資收入"]
ENTITY_TYPE = ["交通費用", "精神慰撫", "醫療費", "收入", "年齡", "其他費用"]
REGULARIZED_TOKEN = ["\n", " ", "\u3000"]
COLUMN_NAME_OF_JSON_CONTENT = "jfull_compress"
REMAIN_KEYS = ["text"]
KEYS_MAPPING_TO_CSV_TABLE = {
    "start": "uie_result_start_index",
    "end": "uie_result_end_index",
    "probability": "uie_result_probability",
}
LABEL_STUDIO_TEMPLATE = {
    "id": 2162,
    "data": {"text": "範例文本", "jid": "範例JID"},
    "annotations": [
        {
            "id": 2192,
            "created_username": " YOUR_MAIL@gmail.com, 1",
            "created_ago": "3 weeks, 1 day",
            "completed_by": {
                "id": 1,
                "first_name": "",
                "last_name": "",
                "avatar": None,
                "email": "範例信箱",
                "initials": "er",
            },
            "result": [
                {
                    "value": {"start": 959, "end": 961, "text": "死亡", "labels": ["受有傷害"]},
                    "id": "EaNTJc9Kw0",
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "origin": "manual",
                }
            ],
            "was_cancelled": False,
            "ground_truth": False,
            "created_at": "2023-06-12T08:22:14.836309Z",
            "updated_at": "2023-06-12T08:41:44.954711Z",
            "lead_time": 1284.776,
            "task": 2162,
            "project": 30,
            "parent_prediction": None,
            "parent_annotation": None,
        }
    ],
    "predictions": [],
}


def load_config(yaml_file):
    with open(yaml_file, "r") as f:
        logger.info(f"Loading the config file in {os.path.abspath(f.name)}...")
        data = yaml.load(f, Loader=yaml.Loader)
    return data


@information_extraction_app.command("dataprep")
def convert(config_file: str = os.path.join(BASE_CONFIG_PATH, "dataprep.yaml")):
    from .dataprep import main

    args = load_config(config_file)

    main(args=args)


@information_extraction_app.command("eval")
def eval(config_file: str = os.path.join(BASE_CONFIG_PATH, "eval.yaml")):
    from .eval import main

    args = load_config(config_file)

    main(args=args)


@information_extraction_app.command("train")
def train(config_file: str = os.path.join(BASE_CONFIG_PATH, "train.yaml")):
    from .train import main

    args = load_config(config_file)

    main(args=args)


@information_extraction_app.command("infer")
def infer(config_file: str = os.path.join(BASE_CONFIG_PATH, "infer.yaml")):
    from .infer import main

    args = load_config(config_file)

    main(args=args)
