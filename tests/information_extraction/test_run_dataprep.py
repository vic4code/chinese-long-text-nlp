import os
import shutil

import pytest

from src.judge.information_extraction.dataprep import main as main_convert
from tests.information_extraction.conftest import is_data_valid_for_uie_input_format


class TestDistinctDataLength:
    def test_convert_success_when_normal_case(self, convert_args):
        # Given
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_normal_case_test.json"
        convert_args["is_regularize_data"] = False

        # When
        main_convert(args=convert_args)

        # Then
        train_path, dev_path, test_path = (
            os.path.join(convert_args["save_dir"], data_type) for data_type in ["train.txt", "dev.txt", "test.txt"]
        )
        assert os.path.exists(train_path)
        assert os.path.exists(dev_path)
        assert os.path.exists(test_path)
        assert is_data_valid_for_uie_input_format(data_path=train_path)
        assert is_data_valid_for_uie_input_format(data_path=dev_path)
        assert is_data_valid_for_uie_input_format(data_path=test_path)

    def test_convert_success_when_only_one_data(self, convert_args):
        # Given
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_one_data_test.json"
        convert_args["is_regularize_data"] = False

        # When
        main_convert(args=convert_args)

        # Then
        train_path = os.path.join(convert_args["save_dir"], "train.txt")
        assert is_data_valid_for_uie_input_format(data_path=train_path)

    def test_convert_fail_when_no_data_then_raise_valueerror(self, convert_args):
        # Given
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_no_data_test.json"
        convert_args["is_regularize_data"] = False

        # When
        with pytest.raises(ValueError) as error:
            main_convert(args=convert_args)

        # Then
        expected_error = f"Data not found in file: {convert_args['labelstudio_file']}"
        assert str(error.value) == expected_error

    def teardown_method(self):
        # Delete convert output
        clean_path = "./data/modeling/information_extraction/model_train/"
        if os.path.exists(clean_path):
            shutil.rmtree(clean_path)
            os.mkdir(clean_path)


class TestDirtyDataForRegularize:
    def test_convert_success_given_regularize_data_when_normal_case(self, convert_args):
        # Given
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_normal_case_test.json"
        convert_args["is_regularize_data"] = True

        # When
        main_convert(args=convert_args)

        # Then
        train_path, dev_path, test_path = (
            os.path.join(convert_args["save_dir"], data_type) for data_type in ["train.txt", "dev.txt", "test.txt"]
        )
        assert os.path.exists(train_path)
        assert os.path.exists(dev_path)
        assert os.path.exists(test_path)
        assert is_data_valid_for_uie_input_format(data_path=train_path)
        assert is_data_valid_for_uie_input_format(data_path=dev_path)
        assert is_data_valid_for_uie_input_format(data_path=test_path)

    def test_convert_success_given_regularize_data_when_dirty_case(self, convert_args):
        # Given
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_dirty_data_test.json"
        convert_args["is_regularize_data"] = True
        convert_args["split_ratio"] = [0.4, 0.3, 0.3]

        # When
        main_convert(args=convert_args)

        # Then
        train_path, dev_path, test_path = (
            os.path.join(convert_args["save_dir"], data_type) for data_type in ["train.txt", "dev.txt", "test.txt"]
        )
        assert os.path.exists(train_path)
        assert os.path.exists(dev_path)
        assert os.path.exists(test_path)
        assert is_data_valid_for_uie_input_format(data_path=train_path)
        assert is_data_valid_for_uie_input_format(data_path=dev_path)
        assert is_data_valid_for_uie_input_format(data_path=test_path)

    def test_convert_success_given_regularize_data_when_strange_case(self, convert_args):
        # Given
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_strange_case_test.json"
        convert_args["is_regularize_data"] = True

        # When
        main_convert(args=convert_args)

        # Then
        train_path = os.path.join(convert_args["save_dir"], "train.txt")
        assert is_data_valid_for_uie_input_format(data_path=train_path)

    def teardown_method(self):
        # Delete convert output
        clean_path = "./data/modeling/information_extraction/model_train/"
        if os.path.exists(clean_path):
            shutil.rmtree(clean_path)
            os.mkdir(clean_path)
