import os
import shutil

from src.judge.text_classification.utc.dataprep import main as main_dataprep
from src.judge.text_classification.utc.train import main as main_train


class TrainPassCriterion:
    def check_if_train_result_exist(self, train_result_path):
        assert os.path.exists(train_result_path)
        assert os.path.exists(os.path.join(train_result_path, "all_results.json"))
        assert os.path.exists(os.path.join(train_result_path, "model_state.pdparams"))
        assert os.path.exists(os.path.join(train_result_path, "vocab.txt"))
        assert os.path.exists(os.path.join(train_result_path, "training_args.bin"))

    def teardown_method(self):
        # # Delete convert output
        if os.path.exists("./tests/text_classification/data/model_train/utc"):
            shutil.rmtree("./tests/text_classification/data/model_train/utc")

        # Delete train output
        if os.path.exists("./models/text_classification/utc/checkpoints/model_best"):
            shutil.rmtree("./models/text_classification/utc/checkpoints/model_best")


class TestBasic(TrainPassCriterion):
    def test_train_success_when_only_one_data(self, label_studio_args, train_args):
        # Given
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/utc/one_data_test.json"
        train_args["training_args"]["num_train_epochs"] = 0.02
        train_args["training_args"]["logging_steps"] = 1

        main_dataprep(args=label_studio_args)

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

    def test_train_success_when_normal_data(self, label_studio_args, train_args):
        # Given
        label_studio_args["label_studio_file"] = "./tests/text_classification/data/label_studio/utc/normal_case_test.json"
        train_args["training_args"]["num_train_epochs"] = 0.003
        train_args["training_args"]["logging_steps"] = 1
        main_dataprep(args=label_studio_args)

        # When
        main_train(args=train_args)

        # Then
        self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])
