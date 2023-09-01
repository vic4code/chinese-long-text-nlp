import os
import shutil

import pytest

from src.judge.text_classification.uie.dataprep import main as main_dataprep
from src.judge.text_classification.uie.train import main as main_train

from .conftest import write_dummy_data_for_model_input


class TrainPassCriterion:
    def check_if_train_result_exist(self, train_result_path):
        assert os.path.exists(train_result_path)
        assert os.path.exists(os.path.join(train_result_path, "all_results.json"))
        assert os.path.exists(os.path.join(train_result_path, "model_state.pdparams"))
        assert os.path.exists(os.path.join(train_result_path, "vocab.txt"))
        assert os.path.exists(os.path.join(train_result_path, "training_args.bin"))

    def teardown_method(self):
        # # Delete convert output
        if os.path.exists("./tests/text_classification/data/model_train/uie"):
            shutil.rmtree("./tests/text_classification/data/model_train/uie")

        # Delete train output
        if os.path.exists("./models/text_classification/uie/checkpoints/model_best"):
            shutil.rmtree("./models/text_classification/uie/checkpoints/model_best")


@pytest.mark.take_long_time
class TestBasic(TrainPassCriterion):
    def test_train_success_when_only_one_data(self, label_studio_args, train_args):
        # Given
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/uie/one_data_test.json"
        train_args["training_args"]["num_train_epochs"] = 0.02
        train_args["training_args"]["logging_steps"] = 1
        main_dataprep(args=label_studio_args)

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_success_when_normal_data(self, label_studio_args, train_args):
        # Given
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/uie/normal_case_test.json"
        train_args["training_args"]["num_train_epochs"] = 0.003
        train_args["training_args"]["logging_steps"] = 1
        main_dataprep(args=label_studio_args)

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])


@pytest.mark.take_long_time
class TestVariousLabel(TrainPassCriterion):
    def test_train_success_when_label_length_large(self, train_args):
        # Given
        dummy_data = {
            "content": "110元臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月",
            "result_list": [{"text": "110元臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月", "start": 0, "end": 69}],
            "prompt": "肇事過失責任比例",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_name="train.txt")
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_name="dev.txt")
        train_args["training_args"]["num_train_epochs"] = 1
        train_args["training_args"]["logging_steps"] = 1

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_success_when_label_in_begin_of_content(self, train_args):
        # Given
        dummy_data = {
            "content": "110元臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月",
            "result_list": [{"text": "110元", "start": 0, "end": 4}],
            "prompt": "肇事過失責任比例",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_name="train.txt")
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_name="dev.txt")
        train_args["training_args"]["num_train_epochs"] = 1
        train_args["training_args"]["logging_steps"] = 1

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_success_when_label_in_end_of_content(self, train_args):
        # Given
        dummy_data = {
            "content": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月110元",
            "result_list": [{"text": "110元", "start": 65, "end": 69}],
            "prompt": "肇事過失責任比例",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_name="train.txt")
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_name="dev.txt")
        train_args["training_args"]["num_train_epochs"] = 1
        train_args["training_args"]["logging_steps"] = 1

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])
