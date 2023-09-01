import json
import os

import pytest


@pytest.fixture
def label_studio_args():
    return {
        "label_studio_file": "./tests/text_classification/data/label_studio/utc/label_studio_for_normal_case_test.json",
        "save_dir": "./tests/text_classification/data/model_train/utc",
        "splits": [0.8, 0.1, 0.1],  # [train, dev, test]
        "text_separator": "\t",
        "label_file": "./tests/text_classification/data/label_studio/utc/labels.txt",
        "is_shuffle": True,
        "seed": 1000,
    }


@pytest.fixture(name="train_args")
def train_args():
    return {
        "model_args": {"model_name_or_path": "utc-base"},
        "data_args": {"dataset_path": "./tests/text_classification/data/model_train/utc", "train_file": "train.txt", "dev_file": "dev.txt"},
        "training_args": {
            "output_dir": "./models/text_classification/utc/checkpoints/model_best",
            "max_seq_length": 768,
            "overwrite_output_dir": True,
            "do_train": True,
            "do_eval": True,
            "do_export": True,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "num_train_epochs": 0.1,
            "lr_scheduler_type": "linear",
            "warmup_ratio": 0.0,
            "warmup_steps": 0,
            "logging_steps": 5,
            "save_strategy": "steps",
            "save_steps": 200,
            "eval_steps": 200,
            "save_total_limit": 1,
            "seed": 30678,
            "device": "gpu",  # or "cpu"
            "disable_tqdm": True,
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_f1",
            "resume_from_checkpoint": None,
            "save_plm": True,
        },
    }


@pytest.fixture
def eval_args():
    return {
        "model_args": {"model_name_or_path": "utc-base", "checkpoint_path": "./models/text_classification/utc/checkpoints/model_best"},
        "data_args": {"test_path": "./tests/text_classification/data/model_train/utc/test.txt", "threshold": 0.5},
        "eval_args": {
            "output_dir": "./models/text_classification/utc/checkpoints/model_best/test_results",
            "max_seq_length": 768,
            "seed": 30678,
            "device": "gpu",  # or "cpu"
            "disable_tqdm": True,
            "metric_for_best_model": "macro_f1",
            "freeze_dropout": True,
            "save_plm": True,
            "freeze_plm": True,
        },
    }


@pytest.fixture
def infer_args():
    return {
        "model_args": {
            "model_name_or_path": "utc-base",
            "checkpoint_path": "./models/text_classification/utc/checkpoints/model_best",
        },
        "data_args": {
            "data_file_to_inference": "../data/model_infer/uie_processed_data.json",
            "label_file": "../data/text_classification/label_studio/text_classification_labels.txt",
            "threshold": 0.5,
        },
        "infer_args": {
            "output_dir": "./models/text_classification/utc/inference_results",
            "max_seq_length": 768,
            "per_device_eval_batch_size": 1,
            "seed": 30678,
            "device": "gpu",  # or "cpu"
            "freeze_dropout": True,
            "freeze_plm": True,
        },
    }


def is_data_valid_for_utc_input_format(data_path):
    is_valid = True
    with open(data_path, "r", encoding="utf8") as f:
        json_list = f.readlines()
        for line in [json_list[0], json_list[-1], json_list[int(len(json_list) / 2)]]:
            json_line = json.loads(line)
            is_valid = is_valid and (list(json_line.keys()) == ["text_a", "text_b", "question", "choices", "labels"])
    return is_valid


def write_dummy_data_for_model_input(dummy_data, write_path="./tests/text_classification/data/model_train/utc", write_name="train.txt"):
    if not os.path.exists(write_path):
        os.mkdir(write_path)

    with open(os.path.join(write_path, write_name), "w", encoding="utf8") as f:
        out_data = json.dumps(dummy_data, ensure_ascii=False)
        f.write(out_data + "\n")
