import os
import shutil

from src.judge.information_extraction.dataprep import main as main_convert
from src.judge.information_extraction.infer import main as main_infer
from src.judge.information_extraction.train import main as main_train


def test_setup(train_args, convert_args):
    convert_args["labelstudio_file"] = "./tests/information_extraction/data/convert_input_data/labelstudio_for_normal_case_test.json"
    main_convert(convert_args)

    train_args["training_args"]["num_train_epochs"] = 0.005
    train_args["training_args"]["logging_steps"] = 1
    main_train(args=train_args)


class TestJsonTypeInput:
    def setup_class(self):
        self.data_path = "./tests/information_extraction/data/inference_input_data"
        self.normal_data = "inference_json_type_normal_data.json"
        self.single_word_data = "inference_json_type_single_word_data.json"
        self.no_features_data = "inference_json_type_no_features_data.json"

    def basic_test(self, infer_args, data_name=None):
        # Given
        data_name = data_name if data_name else self.normal_data
        infer_args["data_args"]["data_file"] = os.path.join(self.data_path, data_name)

        # When
        main_infer(args=infer_args)

        # Then
        infer_out_file = os.listdir("./reports/information_extraction/inference_results")
        assert len(infer_out_file) == 4  # Output File Exist

    def test_success_given_not_regular_when_normal_case(self, infer_args):
        infer_args["data_args"]["is_regularize_data"] = False
        self.basic_test(infer_args=infer_args)

    def test_success_given_regular_when_normal_case(self, infer_args):
        infer_args["data_args"]["is_regularize_data"] = True
        self.basic_test(infer_args=infer_args)

    def test_success_when_no_taskpath(self, infer_args):
        infer_args["taskflow_args"]["task_path"] = None
        self.basic_test(infer_args=infer_args)

    def test_success_when_json_data_with_no_features_but_jfull_compress(self, infer_args):
        self.basic_test(infer_args=infer_args, data_name=self.no_features_data)

    def test_success_when_data_length_only_one_word(self, infer_args):
        self.basic_test(infer_args=infer_args, data_name=self.single_word_data)

    def test_success_when_select_strategy_is_all(self, infer_args):
        infer_args["strategy_args"]["select_strategy"] = "all"
        infer_args["strategy_args"]["select_key"] = ["text", "probability"]
        self.basic_test(infer_args=infer_args)

    def test_success_when_select_strategy_is_threshold(self, infer_args):
        infer_args["strategy_args"]["select_strategy"] = "threshold"
        infer_args["strategy_args"]["select_key"] = ["text", "start", "end"]
        self.basic_test(infer_args=infer_args)

    def test_success_when_select_strategy_is_max(self, infer_args):
        infer_args["strategy_args"]["select_strategy"] = "max"
        infer_args["strategy_args"]["select_key"] = ["end", "probability"]
        self.basic_test(infer_args=infer_args)

    def teardown_method(self):
        # Delete infer output
        if os.path.exists("./reports/information_extraction/inference_results"):
            shutil.rmtree("./reports/information_extraction/inference_results")


class TestTxtTypeInput(TestJsonTypeInput):
    def setup_class(self):
        self.data_path = "./tests/information_extraction/data/inference_input_data"
        self.normal_data = "inference_txt_type_normal_data.txt"
        self.single_word_data = "inference_txt_type_single_data.txt"

    def test_success_when_json_data_with_no_features_but_jfull_compress(self):
        pass


def test_teardown():
    # Delete convert output
    if os.path.exists("./data/modeling/information_extraction/model_train/"):
        shutil.rmtree("./data/modeling/information_extraction/model_train/")
        os.mkdir("./data/modeling/information_extraction/model_train/")

    # Delete train output
    if os.path.exists("./models/information_extraction/checkpoint/model_best"):
        shutil.rmtree("./models/information_extraction/checkpoint/model_best")
