import json
import os
import shutil

from src.judge.text_classification.uie.dataprep import main as main_dataprep
from src.judge.text_classification.uie.eval import main as main_eval
from src.judge.text_classification.uie.train import main as main_train

from .conftest import write_dummy_data_for_model_input


class TestEval:
    def test_setup_class(self, train_args, label_studio_args):
        main_dataprep(label_studio_args)
        train_args["training_args"]["num_train_epochs"] = 0.003
        train_args["training_args"]["logging_steps"] = 1
        main_train(args=train_args)

    def test_success_when_normal_case(self, eval_args):
        # Given
        eval_args["test_path"] = "./tests/text_classification/data/model_train/uie/test.txt"

        # When
        main_eval(args=eval_args)

        # Then
        eval_results_path = "./models/text_classification/uie/checkpoints/model_best/test_results/test_metrics.json"
        assert os.path.exists(eval_results_path)
        with open(eval_results_path, "r", encoding="utf8") as f:
            eval_results = json.loads(f.read())
            assert list(eval_results.keys()) == ["precision", "recall", "f1"]
            assert isinstance(eval_results["precision"], float) and eval_results["precision"] >= 0.0 and eval_results["precision"] <= 1
            assert isinstance(eval_results["recall"], float) and eval_results["recall"] >= 0.0 and eval_results["recall"] <= 1
            assert isinstance(eval_results["f1"], float) and eval_results["f1"] >= 0.0 and eval_results["f1"] <= 1

    def test_success_when_only_one_data(self, eval_args):
        # Given
        dummy_data = {
            "content": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月",
            "result_list": [{"text": "110元", "start": 53, "end": 57}],
            "prompt": "年齡",
        }
        write_dummy_data_for_model_input(dummy_data, write_path="./tests/text_classification/data/model_train/uie", write_name="test.txt")
        eval_args["test_path"] = "./tests/text_classification/data/model_train/uie/test.txt"

        # When
        main_eval(args=eval_args)

        # Then
        eval_results_path = "./models/text_classification/uie/checkpoints/model_best/test_results/test_metrics.json"
        assert os.path.exists(eval_results_path)
        with open(eval_results_path, "r", encoding="utf8") as f:
            eval_results = json.loads(f.read())
            assert isinstance(eval_results["precision"], float) and eval_results["precision"] >= 0.0 and eval_results["precision"] <= 1
            assert isinstance(eval_results["recall"], float) and eval_results["recall"] >= 0.0 and eval_results["recall"] <= 1
            assert isinstance(eval_results["f1"], float) and eval_results["f1"] >= 0.0 and eval_results["f1"] <= 1

    def teardown_class(self):
        # Delete convert output
        if os.path.exists("./tests/text_classification/data/model_train/uie"):
            shutil.rmtree("./tests/text_classification/data/model_train/uie")

        # Delete train output
        if os.path.exists("./models/text_classification/uie/checkpoints/model_best"):
            shutil.rmtree("./models/text_classification/uie/checkpoints/model_best")

    def teardown_method(self):
        # Delete eval output
        if os.path.exists("./models/text_classification/uie/checkpoints/model_best/test_results"):
            shutil.rmtree("./models/text_classification/uie/checkpoints/model_best/test_results")
