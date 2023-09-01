import json
import os
import shutil

import pytest

from src.judge.information_extraction.dataprep import main as main_convert
from src.judge.information_extraction.eval import main as main_eval
from src.judge.information_extraction.train import main as main_train
from tests.information_extraction.conftest import write_dummy_data_for_model_input


@pytest.mark.take_long_time
class TestEval:
    def test_setup_class(self, train_args, convert_args):
        convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_normal_case_test.json"
        convert_args["save_dir"] = "./data/modeling/information_extraction/model_train/normal_case/"
        main_convert(convert_args)

        train_args["training_args"]["num_train_epochs"] = 0.003
        train_args["training_args"]["logging_steps"] = 1
        train_args["data_args"]["dataset_path"] = "./data/modeling/information_extraction/model_train/normal_case/"
        main_train(args=train_args)

    def test_success_given_eval_by_group_when_normal_case(self, eval_args):
        # Given
        eval_args["dev_file"] = "./data/modeling/information_extraction/model_train/normal_case/test.txt"
        eval_args["is_eval_by_class"] = True

        # When
        main_eval(args=eval_args)

        # Then
        eval_results_path = "./models/information_extraction/checkpoint/model_best/test_results/test_metrics.json"
        assert os.path.exists(eval_results_path)
        with open(eval_results_path, "r", encoding="utf8") as f:
            eval_results = json.loads(f.read())
            assert list(eval_results.keys()) == ["精神慰撫金額", "醫療費用", "薪資收入", "total"]
            assert isinstance(eval_results["精神慰撫金額"]["precision"], float) and eval_results["精神慰撫金額"]["precision"] >= 0.0
            assert isinstance(eval_results["醫療費用"]["recall"], float) and eval_results["醫療費用"]["recall"] >= 0.0
            assert isinstance(eval_results["total"]["f1"], float) and eval_results["total"]["f1"] >= 0.0

    def test_success_given_eval_by_all_when_normal_case(self, eval_args):
        # Given
        eval_args["dev_file"] = "./data/modeling/information_extraction/model_train/normal_case/test.txt"
        eval_args["is_eval_by_class"] = False

        # When
        main_eval(args=eval_args)

        # Then
        eval_results_path = "./models/information_extraction/checkpoint/model_best/test_results/test_metrics.json"
        assert os.path.exists(eval_results_path)
        with open(eval_results_path, "r", encoding="utf8") as f:
            eval_results = json.loads(f.read())
            assert list(eval_results.keys()) == ["precision", "recall", "f1"]
            assert isinstance(eval_results["precision"], float) and eval_results["precision"] >= 0.0
            assert isinstance(eval_results["recall"], float) and eval_results["recall"] >= 0.0
            assert isinstance(eval_results["f1"], float) and eval_results["f1"] >= 0.0

    def test_success_when_only_one_data(self, eval_args):
        # Given
        dummy_data = {
            "content": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月",
            "result_list": [{"text": "110元", "start": 53, "end": 57}],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data, write_name="test.txt")
        eval_args["dev_file"] = "./data/modeling/information_extraction/model_train/test.txt"
        eval_args["is_eval_by_class"] = True

        # When
        main_eval(args=eval_args)

        # Then
        eval_results_path = "./models/information_extraction/checkpoint/model_best/test_results/test_metrics.json"
        assert os.path.exists(eval_results_path)
        with open(eval_results_path, "r", encoding="utf8") as f:
            eval_results = json.loads(f.read())
            assert list(eval_results.keys()) == ["精神慰撫金額", "醫療費用", "薪資收入", "total"]
            assert isinstance(eval_results["精神慰撫金額"]["precision"], float) and eval_results["精神慰撫金額"]["precision"] >= 0.0
            assert isinstance(eval_results["醫療費用"]["recall"], float) and eval_results["醫療費用"]["recall"] >= 0.0
            assert isinstance(eval_results["total"]["f1"], float) and eval_results["total"]["f1"] >= 0.0

    def teardown_class(self):
        # Delete convert output
        if os.path.exists("./data/modeling/information_extraction/model_train/"):
            shutil.rmtree("./data/modeling/information_extraction/model_train/")
            os.mkdir("./data/modeling/information_extraction/model_train/")

        # Delete train output
        if os.path.exists("./models/information_extraction/checkpoint/model_best"):
            shutil.rmtree("./models/information_extraction/checkpoint/model_best")

    def teardown_method(self):
        # Delete eval output
        if os.path.exists("./models/information_extraction/checkpoint/model_best/test_results"):
            shutil.rmtree("./models/information_extraction/checkpoint/model_best/test_results")
