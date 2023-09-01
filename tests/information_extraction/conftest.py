import json
import os

import pytest

from src.judge.information_extraction import COLUMN_NAME_OF_JSON_CONTENT, ENTITY_TYPE


@pytest.fixture
def convert_args():
    return {
        "labelstudio_file": "./tests/information_extraction/data/labelstudio_for_e2e_test.json",
        "save_dir": "./data/modeling/information_extraction/model_train/",
        "seed": 1000,
        "split_ratio": [0.8, 0.1, 0.1],
        "is_shuffle": True,
        "is_regularize_data": False,
    }


@pytest.fixture(name="train_args")
def train_args():
    return {
        "model_args": {
            "model_name_or_path": "uie-base",
            "max_seq_len": 256,
        },
        "data_args": {
            "dataset_path": "./data/modeling/information_extraction/model_train/",
            "train_file": "train.txt",
            "dev_file": "dev.txt",
            "test_file": "test.txt",
            "export_model_dir": None,
        },
        "training_args": {
            "output_dir": "./models/information_extraction/checkpoint/model_best",
            "overwrite_output_dir": True,
            "do_train": True,
            "do_eval": False,
            "do_predict": False,
            "do_export": True,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 16,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "num_train_epochs": 0.01,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.0,
            "warmup_steps": 0,
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": 200,
            "eval_steps": 200,
            "save_total_limit": None,
            "seed": 11,
            "device": "gpu",
            "disable_tqdm": True,
            "label_names": ["start_positions", "end_positions"],
            "load_best_model_at_end": False,
            "metric_for_best_model": "eval_f1",
            "resume_from_checkpoint": None,
        },
    }


@pytest.fixture
def eval_args():
    return {
        "model_name_or_path": "./models/information_extraction/checkpoint/model_best",
        "dev_file": "./data/modeling/information_extraction/model_train/test.txt",
        "batch_size": 16,
        "device": "gpu",
        "is_eval_by_class": False,
        "max_seq_len": 256,
    }


@pytest.fixture
def infer_args():
    return {
        "data_args": {
            "data_file": "./data/modeling/information_extraction/model_train/test.txt",
            "save_dir": "./reports/information_extraction/inference_results/",
            "save_name": "inference_results.json",
            "is_regularize_data": True,
            "is_export_labelstudio": True,
            "is_export_csv": True,
            "is_regularize_csv_money": True,
            "text_list": None,
        },
        "taskflow_args": {
            "device_id": 0,
            "precision": "fp32",
            "batch_size": 1,
            "model": "uie-base",
            "task_path": "./models/information_extraction/checkpoint/model_best",
        },
        "strategy_args": {
            "select_strategy": "all",
            "select_strategy_threshold": 0.5,
            "select_key": ["text", "start", "end", "probability"],
        },
    }


@pytest.fixture
def label_studio_template():
    return {
        "id": 2162,
        "data": {COLUMN_NAME_OF_JSON_CONTENT: "範例文本", "jid": "範例JID"},
        "annotations": [
            {
                "id": 2192,
                "created_username": "aaaaa@gmail.com, 1",
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


@pytest.fixture
def uie_dummy_result():
    return [
        {
            "精神慰撫金額": [
                {"text": "150,000元", "start": 5188, "end": 5196, "probability": 0.9977151884524105},
                {"text": "150,000", "start": 5507, "end": 5514, "probability": 0.9481814888050799},
            ],
            "薪資收入": [
                {"text": "80,856", "start": 4452, "end": 4458, "probability": 0.9853195936072368},
            ],
        }
    ]


def is_data_valid_for_uie_input_format(data_path):
    is_valid = True
    with open(data_path, "r", encoding="utf8") as f:
        json_list = f.readlines()
        for line in [json_list[0], json_list[-1], json_list[int(len(json_list) / 2)]]:
            json_line = json.loads(line)
            is_valid = is_valid and (list(json_line.keys()) == ["content", "result_list", "prompt"])
            is_valid = is_valid and (json_line["prompt"] in ENTITY_TYPE)
            if json_line["result_list"]:
                test_target = json_line["result_list"][0]
                is_valid = is_valid and (json_line["content"][test_target["start"] : test_target["end"]] == test_target["text"])
    return is_valid


def write_dummy_data_for_label_studio_output_format(file, write_path):
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    with open(os.path.join(write_path, "dummy_data.json"), "w", encoding="utf8") as f:
        jsonString = json.dumps(file, ensure_ascii=False)
        f.write(jsonString)


def write_dummy_data_for_model_input(dummy_data, write_path="./data/modeling/information_extraction/model_train/", write_name="train.txt"):
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    with open(os.path.join(write_path, write_name), "w", encoding="utf8") as f:
        out_data = json.dumps(dummy_data, ensure_ascii=False)
        f.write(out_data + "\n")
