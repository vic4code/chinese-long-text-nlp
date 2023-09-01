import os
import shutil

import pytest

from src.judge.information_extraction.dataprep import main as main_convert
from src.judge.information_extraction.train import main as main_train
from tests.information_extraction.conftest import write_dummy_data_for_model_input


class TrainPassCriterion:
    def check_if_train_result_exist(self, train_result_path):
        assert os.path.exists(train_result_path)
        assert os.path.exists(os.path.join(train_result_path, "all_results.json"))
        assert os.path.exists(os.path.join(train_result_path, "model_state.pdparams"))
        assert os.path.exists(os.path.join(train_result_path, "vocab.txt"))
        assert os.path.exists(os.path.join(train_result_path, "training_args.bin"))

    def teardown_method(self):
        # Delete convert output
        if os.path.exists("./data/modeling/information_extraction/model_train/"):
            shutil.rmtree("./data/modeling/information_extraction/model_train/")
            os.mkdir("./data/modeling/information_extraction/model_train/")

        # Delete train output
        if os.path.exists("./models/information_extraction/checkpoint/model_best"):
            shutil.rmtree("./models/information_extraction/checkpoint/model_best")


@pytest.mark.take_long_time
class TestBasic(TrainPassCriterion):
    def test_train_success_when_only_one_data(self, convert_args, train_args):
        # Given
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_one_data_test.json"
        train_args["training_args"]["num_train_epochs"] = 0.02
        train_args["training_args"]["logging_steps"] = 1
        main_convert(args=convert_args)

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_success_when_normal_data(self, convert_args, train_args):
        # Given
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_normal_case_test.json"
        train_args["training_args"]["num_train_epochs"] = 0.003
        train_args["training_args"]["logging_steps"] = 1
        main_convert(args=convert_args)

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_success_when_doeval_dopredict(self, convert_args, train_args):
        # Given
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_normal_case_test.json"
        main_convert(args=convert_args)

        # When
        train_args["training_args"]["logging_steps"] = 1
        train_args["training_args"]["num_train_epochs"] = 0.05
        train_args["training_args"]["save_steps"] = 20
        train_args["training_args"]["eval_steps"] = 20
        train_args["training_args"]["do_eval"] = True
        train_args["training_args"]["do_predict"] = True
        train_args["training_args"]["load_best_model_at_end"] = True
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])


@pytest.mark.take_long_time
class TestContentLength(TrainPassCriterion):
    def setup_class(self):
        self.dummy_data = {
            "content": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月",
            "result_list": [{"text": "110元", "start": 53, "end": 57}],
            "prompt": "醫療費用",
        }

    def test_train_success_when_text_length_less_than_100(self, train_args):
        # Given
        write_dummy_data_for_model_input(dummy_data=self.dummy_data)
        train_args["training_args"]["num_train_epochs"] = 1
        train_args["training_args"]["logging_steps"] = 1

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_success_when_text_length_equal_to_max_length(self, train_args):
        # Given
        write_dummy_data_for_model_input(dummy_data=self.dummy_data)
        train_args["training_args"]["num_train_epochs"] = 1
        train_args["training_args"]["logging_steps"] = 1
        train_args["model_args"]["max_seq_len"] = len(self.dummy_data["content"])

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_success_when_max_length_smallest(self, train_args):
        # Given
        write_dummy_data_for_model_input(dummy_data=self.dummy_data)
        train_args["training_args"]["num_train_epochs"] = 0.2
        train_args["training_args"]["logging_steps"] = 1
        smallest_scale = self.dummy_data["result_list"][0]["end"] - self.dummy_data["result_list"][0]["start"]
        train_args["model_args"]["max_seq_len"] = len(self.dummy_data["prompt"]) + 3 + smallest_scale

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_fail_when_max_length_less_than_smallest(self, train_args):
        # Given
        write_dummy_data_for_model_input(dummy_data=self.dummy_data)
        train_args["training_args"]["num_train_epochs"] = 0.2
        train_args["training_args"]["logging_steps"] = 1
        smallest_scale = self.dummy_data["result_list"][0]["end"] - self.dummy_data["result_list"][0]["start"]
        train_args["model_args"]["max_seq_len"] = len(self.dummy_data["prompt"]) + 3 + smallest_scale - 1

        # When
        with pytest.raises(ValueError) as error:
            main_train(args=train_args)

        # Then
        expected_error = (
            f"end - start > max_content_len, {self.dummy_data['result_list'][0]['end']} - "
            f"{self.dummy_data['result_list'][0]['start']} > {smallest_scale - 1}, "
            "Please adjust label index or scale up max_seq_len"
        )
        assert str(error.value) == expected_error


@pytest.mark.take_long_time
class TestVariousLabel(TrainPassCriterion):
    def test_train_success_when_label_length_large(self, train_args):
        # Given
        dummy_data = {
            "content": "110元臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月",
            "result_list": [{"text": "110元臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月", "start": 0, "end": 69}],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data)
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
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data)
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
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data)
        train_args["training_args"]["num_train_epochs"] = 1
        train_args["training_args"]["logging_steps"] = 1

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_success_when_content_full_of_label(self, train_args):
        # Given
        dummy_data = {
            "content": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月110元",
            "result_list": [
                {"text": "臺灣", "start": 0, "end": 2},
                {"text": "苗栗地方法院", "start": 2, "end": 8},
                {"text": "事裁定110年度", "start": 9, "end": 17},
                {"text": "苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定", "start": 17, "end": 51},
                {"text": "賠償110元，民國50年三月", "start": 51, "end": 65},
                {"text": "110元", "start": 65, "end": 69},
            ],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data)
        train_args["training_args"]["num_train_epochs"] = 1
        train_args["training_args"]["logging_steps"] = 1
        train_args["model_args"]["max_seq_len"] = 41  # minimum value

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_fail_when_label_text_is_not_equal_to_index(self, train_args):
        # Given
        dummy_data = {
            "content": "110元臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月",
            "result_list": [{"text": "110元", "start": 1, "end": 5}],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data)
        train_args["training_args"]["num_train_epochs"] = 1
        train_args["training_args"]["logging_steps"] = 1

        # When
        with pytest.raises(ValueError) as error:
            main_train(args=train_args)

        # Then
        wrong_data = dummy_data["content"][dummy_data["result_list"][0]["start"] : dummy_data["result_list"][0]["end"]]
        expected_error = f"adjust error. adjust_data: {wrong_data}, true_data: {dummy_data['result_list'][0]['text']}."
        assert str(error.value) == expected_error
