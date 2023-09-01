import os
import shutil

import pytest

from src.judge.text_classification.uie.dataprep import main as main_dataprep

from .conftest import is_data_valid_for_uie_input_format


class TestDistinctDataLength:
    def test_convert_success_when_normal_case(self, label_studio_args):
        # Given
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/uie/normal_case_test.json"

        # When
        main_dataprep(args=label_studio_args)

        # Then
        train_path, dev_path, test_path = (
            os.path.join(label_studio_args["save_dir"], data_type) for data_type in ["train.txt", "dev.txt", "test.txt"]
        )
        assert os.path.exists(train_path)
        assert os.path.exists(dev_path)
        assert os.path.exists(test_path)
        assert is_data_valid_for_uie_input_format(data_path=train_path)
        assert is_data_valid_for_uie_input_format(data_path=dev_path)
        assert is_data_valid_for_uie_input_format(data_path=test_path)

    def test_convert_success_when_only_one_data(self, label_studio_args):
        # Given
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/uie/one_data_test.json"

        # When
        main_dataprep(args=label_studio_args)

        # Then
        train_path = os.path.join(label_studio_args["save_dir"], "train.txt")
        assert is_data_valid_for_uie_input_format(data_path=train_path)

    def test_convert_fail_when_no_data_then_raise_valueerror(self, label_studio_args):
        # Given
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/uie/no_data_test.json"

        # When
        with pytest.raises(ValueError) as error:
            main_dataprep(args=label_studio_args)

        # Then
        expected_error = f"Data not found in file: {label_studio_args['label_studio_file']}"
        print(expected_error)
        assert str(error.value) == expected_error

    def teardown_method(self):
        # Delete convert output
        clean_path = "./data/information_extraction/model_input"
        if os.path.exists(clean_path):
            shutil.rmtree(clean_path)
