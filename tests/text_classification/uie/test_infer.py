import os
import shutil

import pytest

from src.judge.text_classification.uie.dataprep import main as main_dataprep
from src.judge.text_classification.uie.infer import main as main_infer
from src.judge.text_classification.uie.train import main as main_train


class TestInfer:
    def test_setup_class(self, train_args, label_studio_args):
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/uie/normal_case_test.json"
        main_dataprep(label_studio_args)

        train_args["training_args"]["num_train_epochs"] = 0.003
        train_args["training_args"]["logging_steps"] = 1
        main_train(args=train_args)

    def test_success_when_utc_train_input(self, infer_args):
        # Given
        infer_args["data_file_or_path_to_inference"] = "./tests/text_classification/data/model_infer/uie/utc_train"

        # When
        main_infer(args=infer_args)

        # Then
        infer_out_files = os.listdir("./tests/text_classification/data/model_infer/uie/utc_train")

        assert "train.txt" in infer_out_files
        assert "dev.txt" in infer_out_files
        assert "test.txt" in infer_out_files

    def test_success_when_verdict_json_input(self, infer_args):
        # Given
        infer_args["data_file_or_path_to_inference"] = "./tests/text_classification/data/model_infer/uie/inference_normal_data.json"

        # When
        main_infer(args=infer_args)

        # Then
        assert os.path.isfile(os.path.join("./tests/text_classification/data/model_infer/uie", "uie_processed_data.json"))

    def test_fail_when_invalid_input_then_raise_valueerror(self, infer_args):
        # Given
        infer_args["data_file_or_path_to_inference"] = "asdfasder/uzxcvie/inference_normal_data.json"

        # When
        with pytest.raises(ValueError) as error:
            main_infer(args=infer_args)

        # Then
        expected_error = (
            "Invalid file or path. The input json file must be exported from CA verdict database and with 'jfull_compress' included."
            "If a path is given, it must include train.txt, dev.txt, test.txt files which are generated from utc_dataprep."
        )
        assert str(error.value) == expected_error

    def teardown_method(self):
        # Delete infer output
        if os.path.exists("./tests/text_classification/data/model_infer/uie/uie_processed_data.json"):
            os.remove("./tests/text_classification/data/model_infer/uie/uie_processed_data.json")

        if os.path.exists("./tests/text_classification/data/model_infer/uie/utc_train/uie_preprocessed"):
            shutil.rmtree("./tests/text_classification/data/model_infer/uie/utc_train/uie_preprocessed")


def test_teardown():
    # Delete dataprep output
    if os.path.exists("./tests/text_classification/data/model_train/uie"):
        shutil.rmtree("./tests/text_classification/data/model_train/uie")

    # Delete train output
    if os.path.exists("./reports/text_classification/uie/checkpoints/model_best"):
        shutil.rmtree("./reports/text_classification/uie/checkpoints/model_best")
