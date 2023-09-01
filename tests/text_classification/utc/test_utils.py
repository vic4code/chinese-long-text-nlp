# import os
# import shutil

# import pytest

# from src.judge.text_classification.dataprep import main as main_convert
# from src.judge.text_classification.train import main as main_train
# from tests.text_classification.conftest import write_dummy_data_for_model_input


# class TestUtils(TrainPassCriterion):
#     def test_read_inference_dataset_success(self, convert_args, train_args):
#         # Given
#         convert_args["labelstudio_file"] = "./tests/text_classification/data/labelstudio_for_one_data_test.json"
#         train_args["training_args"]["num_train_epochs"] = 0.02
#         train_args["training_args"]["logging_steps"] = 1
#         main_convert(args=convert_args)

#         # When
#         main_train(args=train_args)

#         # Then
#         self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])

#     def test_read_inference_dataset_fail(self, convert_args, train_args):
#         # Given
#         convert_args["labelstudio_file"] = "./tests/text_classification/data/labelstudio_for_normal_case_test.json"
#         train_args["training_args"]["num_train_epochs"] = 0.003
#         train_args["training_args"]["logging_steps"] = 1
#         main_convert(args=convert_args)

#         # When
#         main_train(args=train_args)

#         # Then
#         self.check_if_train_result_exist(train_result_path=train_args["training_args"]["output_dir"])
