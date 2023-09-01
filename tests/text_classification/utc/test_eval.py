import json
import os
import shutil

from src.judge.text_classification.utc.dataprep import main as main_dataprep
from src.judge.text_classification.utc.eval import main as main_eval
from src.judge.text_classification.utc.train import main as main_train

from .conftest import write_dummy_data_for_model_input


class EvalPassCriterion:
    def check_if_eval_result_exists(self, eval_metrics_path, eval_predictions_path):
        assert os.path.exists(eval_metrics_path)
        assert os.path.exists(eval_predictions_path)

        with open(eval_metrics_path, "r", encoding="utf8") as f:
            eval_metrics = json.loads(f.read())
            assert list(eval_metrics.keys()) == [
                "test_loss",
                "test_eval_micro_f1",
                "test_eval_macro_f1",
                "test_accuracy_score",
                "test_precision_score",
                "test_recall_score",
                "test_runtime",
                "test_samples_per_second",
                "test_steps_per_second",
            ]
            assert isinstance(eval_metrics["test_loss"], float) and eval_metrics["test_loss"] >= 0.0
            assert isinstance(eval_metrics["test_eval_micro_f1"], float) and eval_metrics["test_eval_micro_f1"] >= 0.0
            assert isinstance(eval_metrics["test_eval_macro_f1"], float) and eval_metrics["test_eval_macro_f1"] >= 0.0
            assert isinstance(eval_metrics["test_accuracy_score"], float) and eval_metrics["test_accuracy_score"] >= 0.0
            assert isinstance(eval_metrics["test_precision_score"], float) and eval_metrics["test_precision_score"] >= 0.0
            assert isinstance(eval_metrics["test_recall_score"], float) and eval_metrics["test_recall_score"] >= 0.0
            assert isinstance(eval_metrics["test_runtime"], float) and eval_metrics["test_runtime"] >= 0.0
            assert isinstance(eval_metrics["test_samples_per_second"], float) and eval_metrics["test_samples_per_second"] >= 0.0
            assert isinstance(eval_metrics["test_steps_per_second"], float) and eval_metrics["test_steps_per_second"] >= 0.0

        with open(eval_predictions_path, "r", encoding="utf8") as f:
            for line in f:
                data = json.loads(line)
                assert list(data.keys()) == ["id", "labels", "probs"]


class TestEval(EvalPassCriterion):
    def test_setup_class(self, train_args, label_studio_args):
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/utc/normal_case_test.json"

        main_dataprep(label_studio_args)

        train_args["training_args"]["num_train_epochs"] = 0.003
        train_args["training_args"]["logging_steps"] = 1
        train_args["data_args"]["dataset_path"] = "./tests/text_classification/data/model_train/utc"
        main_train(args=train_args)

    def test_success_given_eval_when_normal_case(self, eval_args):
        # Given
        eval_args["data_args"]["test_path"] = "./tests/text_classification/data/model_train/utc/test.txt"

        # When
        main_eval(args=eval_args)

        # Then
        eval_metrics_path = "./models/text_classification/utc/checkpoints/model_best/test_results/test_metrics.json"
        eval_predictions_path = "./models/text_classification/utc/checkpoints/model_best/test_results/test_predictions.json"

        # Test
        self.check_if_eval_result_exists(eval_metrics_path, eval_predictions_path)

    def test_success_when_only_one_data(self, eval_args):
        # Given
        label_file = "./tests/text_classification/data/label_studio/utc/labels.txt"
        with open(label_file, "r", encoding="utf-8") as fp:
            labels = [x.strip() for x in fp]

        dummy_data = {
            "text_a": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月",
            "text_b": "",
            "question": "",
            "choices": labels,
            "labels": [6, 7, 11],
        }

        write_dummy_data_for_model_input(dummy_data, write_name="test.txt")
        eval_args["data_args"]["test_path"] = "./tests/text_classification/data/model_train/utc/test.txt"

        # When
        main_eval(args=eval_args)

        # Then
        eval_metrics_path = "./models/text_classification/utc/checkpoints/model_best/test_results/test_metrics.json"
        eval_predictions_path = "./models/text_classification/utc/checkpoints/model_best/test_results/test_predictions.json"

        # Test
        self.check_if_eval_result_exists(eval_metrics_path, eval_predictions_path)

    def teardown_class(self):
        # Delete convert output
        if os.path.exists("./tests/text_classification/data/model_train/utc"):
            shutil.rmtree("./tests/text_classification/data/model_train/utc")

        # Delete train output
        if os.path.exists("./models/text_classification/utc/checkpoints/model_best"):
            shutil.rmtree("./models/text_classification/utc/checkpoints/model_best")

    def teardown_method(self):
        # Delete eval output
        if os.path.exists("./models/text_classification/utc/checkpoints/model_best/test_results"):
            shutil.rmtree("./models/text_classification/utc/checkpoints/model_best/test_results")
