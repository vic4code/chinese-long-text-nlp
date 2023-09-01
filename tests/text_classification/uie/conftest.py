import json
import os

import paddle
import pytest
from paddlenlp.utils.log import logger


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    yield
    paddle.device.cuda.empty_cache()


@pytest.fixture
def label_studio_args():
    return {
        "label_studio_file": "./tests/text_classification/data/label_studio/uie/normal_case_test.json",
        "save_dir": "./tests/text_classification/data/model_train/uie",
        "negative_ratio": 5,
        "split_ratio": [0.8, 0.1, 0.1],  # [train, dev, test]
        "task_type": "ext",
        "options": ["正向", "负向"],
        "prompt_prefix": "情感倾向",
        "is_shuffle": True,
        "seed": 1000,
        "separator": "##",
        "schema_lang": "ch",
    }


@pytest.fixture(name="train_args")
def train_args():
    return {
        "model_args": {"model_name_or_path": "uie-base"},
        "data_args": {
            "train_path": "./tests/text_classification/data/model_train/uie/train.txt",
            "dev_path": "./tests/text_classification/data/model_train/uie/dev.txt",
            "max_seq_length": 512,
        },
        "training_args": {
            "output_dir": "./models/text_classification/uie/checkpoints/model_best",
            "overwrite_output_dir": True,
            "do_train": True,
            "do_eval": True,
            "do_predict": True,
            "do_export": True,
            "do_compress": False,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "num_train_epochs": 0.2,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.0,
            "warmup_steps": 0,
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": 200,
            "eval_steps": 200,
            "save_total_limit": None,
            "seed": 1000,
            "device": "gpu",  # or "cpu"
            "disable_tqdm": True,
            "label_names": ['start_positions', 'end_positions'],
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_f1",
            "resume_from_checkpoint": None,
        },
    }


@pytest.fixture
def eval_args():
    return {
        "checkpoint_path": "./models/text_classification/uie/checkpoints/model_best",
        "test_path": "./tests/text_classification/data/model_train/uie/test.txt",
        "output_dir": "./models/text_classification/uie/checkpoints/model_best/test_results",
        "batch_size": 16,
        "device": "gpu",
        "max_seq_length": 768,
        "multilingual": False,
        "schema_lang": "ch",
    }


@pytest.fixture
def infer_args():
    return {
        "label_file": "./tests/text_classification/data/label_studio/utc/labels.txt",
        "data_file_or_path_to_inference": "./tests/text_classification/data/model_infer/uie/inference_normal_data.json",
        "max_seq_length": 768,
        "schema": ["原告年齡", "肇事過失責任比例", "受有傷害"],
        "utc_model_name_or_path": "utc-base",
        "uie_model_name_or_path": "./models/text_classification/uie/checkpoints/model_best",
        "threshold": 0.4,
        "dynamic_adjust_length": True,
    }


@pytest.fixture
def label_studio_template():
    return {
        "id": 2162,
        "data": {"text": "範例文本", "jid": "範例JID"},
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


def is_data_valid_for_uie_input_format(data_path):
    is_valid = True
    with open(data_path, "r", encoding="utf8") as f:
        json_list = f.readlines()
        if json_list:
            for line in [json_list[0], json_list[-1], json_list[int(len(json_list) / 2)]]:
                json_line = json.loads(line)
                is_valid = is_valid and (list(json_line.keys()) == ["content", "result_list", "prompt"])
                if json_line["result_list"]:
                    test_target = json_line["result_list"][0]
                    is_valid = is_valid and (json_line["content"][test_target["start"] : test_target["end"]] == test_target["text"])
        if "dev" in data_path:
            logger.warning(f"Skip the blank {data_path} file")
    return is_valid


def write_dummy_data_for_label_studio_output_format(file, write_path):
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    with open(os.path.join(write_path, "dummy_data.json"), "w", encoding="utf8") as f:
        jsonString = json.dumps(file, ensure_ascii=False)
        f.write(jsonString)


def write_dummy_data_for_model_input(dummy_data, write_path="./tests/text_classification/data/model_train/uie", write_name="train.txt"):
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    with open(os.path.join(write_path, write_name), "w", encoding="utf8") as f:
        out_data = json.dumps(dummy_data, ensure_ascii=False)
        f.write(out_data + "\n")
