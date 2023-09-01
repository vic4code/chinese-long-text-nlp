import os
import shutil

import pytest

from src.judge.text_classification.utc.dataprep import main as main_dataprep
from src.judge.text_classification.utc.infer import main as main_infer
from src.judge.text_classification.utc.train import main as main_train


class TestInfer:
    def test_setup_class(self, train_args, label_studio_args):
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/utc/normal_case_test.json"
        main_dataprep(label_studio_args)

        train_args["training_args"]["num_train_epochs"] = 0.003
        train_args["training_args"]["logging_steps"] = 1
        train_args["data_args"]["dataset_path"] = "./tests/text_classification/data/model_train/utc"
        main_train(args=train_args)

    def test_success_when_normal_case(self, infer_args):
        # Given
        infer_args["data_args"]["data_file_to_inference"] = "./tests/text_classification/data/model_infer/utc/inference_normal_data.json"
        infer_args["data_args"]["label_file"] = "./tests/text_classification/data/label_studio/utc/labels.txt"
        infer_args["infer_args"]["output_dir"] = "./reports/text_classification/utc/inference_results"

        # When
        main_infer(args=infer_args)

        # Then
        assert os.path.isfile("./reports/text_classification/utc/inference_results/inference_results.json")

    def test_fail_when_invalid_label_file_then_raise_valueerror(self, infer_args):
        # Given
        infer_args["data_args"]["label_file"] = ""
        infer_args["data_args"]["data_file_to_inference"] = "./tests/text_classification/data/model_infer/utc/inference_normal_data.json"

        # When
        with pytest.raises(ValueError) as error:
            main_infer(args=infer_args)

        # Then
        expected_error = "Invalid label_file. Please use file with one label per line or set `label_file` with condidate labels."
        assert str(error.value) == expected_error

    def test_fail_when_invalid_inference_data_then_raise_valueerror(self, infer_args):
        # Given
        infer_args["data_args"]["label_file"] = "./tests/text_classification/data/label_studio/utc/labels.txt"
        infer_args["data_args"]["data_file_to_inference"] = ""

        # When
        with pytest.raises(ValueError) as error:
            main_infer(args=infer_args)

        # Then
        expected_error = (
            "Invalid file or path. The input json file must be exported from CA verdict database and with 'jfull_compress' included."
        )
        assert str(error.value) == expected_error

    def teardown_method(self):
        # Delete infer output
        if os.path.exists("./reports/text_classification/utc/inference_results"):
            shutil.rmtree("./reports/text_classification/utc/inference_results")


def test_teardown():
    # Delete dataprep output
    if os.path.exists("./tests/text_classification/data/model_train/utc"):
        shutil.rmtree("./tests/text_classification/data/model_train/utc")

    # Delete train output
    if os.path.exists("./reports/text_classification/utc/checkpoints/model_best"):
        shutil.rmtree("./reports/text_classification/utc/checkpoints/model_best")
