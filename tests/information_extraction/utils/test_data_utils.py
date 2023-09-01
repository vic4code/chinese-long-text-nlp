import os
import shutil

import numpy as np
import pytest
from paddlenlp.transformers import AutoTokenizer

from src.judge.information_extraction.utils.data_utils import convert_to_uie_format, read_data_by_chunk
from tests.information_extraction.conftest import write_dummy_data_for_model_input


class TestReadDataByChunk:
    def setup_method(self):
        self.write_path = "./tests/information_extraction/data/test"
        self.write_name = "test.txt"

    def test_success_when_normal_case(self):
        # Given
        dummy_data = {
            "content": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月110元",
            "result_list": [{"text": "110元", "start": 65, "end": 69}],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_path=self.write_path, write_name=self.write_name)

        # When
        output_generator = read_data_by_chunk(data_path=os.path.join(self.write_path, self.write_name), max_seq_len=512)
        output_data = next(output_generator)
        with pytest.raises(StopIteration) as error:
            next(output_generator)

        # Then
        assert dummy_data == output_data
        assert error.type == StopIteration

    def test_success_when_long_case(self):
        # Given
        prompt = "精神慰撫金額"

        # When
        output_generator = read_data_by_chunk(
            data_path=os.path.join("./tests/information_extraction/data/model_input_data", self.write_name),
            max_seq_len=512,
        )

        # Then
        expected_content_len = 512 - len(prompt) - 3
        expected_first_answer_locate_chunk = 8877 // expected_content_len
        expected_first_answer_locate_start_index = 8877 % expected_content_len
        expected_first_answer_locate_end_index = 8881 % expected_content_len

        chunk = 0
        first_answer_locate = {}
        for i in output_generator:
            if i["result_list"]:
                first_answer_locate = i
                break
            chunk += 1

        assert len(first_answer_locate["content"]) == expected_content_len
        assert chunk == expected_first_answer_locate_chunk
        assert first_answer_locate["result_list"][0]["start"] == expected_first_answer_locate_start_index
        assert first_answer_locate["result_list"][0]["end"] == expected_first_answer_locate_end_index

    def test_success_when_result_cross_chunk(self):
        # Given
        dummy_data = {
            "content": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償999元，民國50年三月110元栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被民事裁定110年度苗簡字第563號"
            "原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原",
            "result_list": [{"text": "110元", "start": 65, "end": 69}],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_path=self.write_path, write_name=self.write_name)

        # When
        output_generator = read_data_by_chunk(data_path=os.path.join(self.write_path, self.write_name), max_seq_len=75)
        first_chunk = next(output_generator)
        second_chunk = next(output_generator)

        # Then
        expected_content_len_when_normal = 75 - len(dummy_data["prompt"]) - 3
        expected_content_len_when_cross_chunk = dummy_data["result_list"][0]["start"]
        expected_adjust_start_index = 0
        expected_adjust_end_index = 4

        assert len(first_chunk["content"]) == expected_content_len_when_cross_chunk
        assert len(second_chunk["content"]) == expected_content_len_when_normal
        assert second_chunk["result_list"][0]["start"] == expected_adjust_start_index
        assert second_chunk["result_list"][0]["end"] == expected_adjust_end_index

    def teardown_method(self):
        if os.path.exists(self.write_path):
            shutil.rmtree(self.write_path)


class TestConvertToUieFormat:
    def setup_method(self):
        self.write_path = "./tests/information_extraction/data/test"
        self.write_name = "test.txt"
        self.tokenizer = AutoTokenizer.from_pretrained("uie-base")

    def basic_test(self, convert_result, expected_text):
        start_index = np.where(np.array(convert_result["start_positions"]) == 1.0)[0][0]
        end_index = np.where(np.array(convert_result["end_positions"]) == 1.0)[0][0]
        converted_text = convert_result["input_ids"][start_index : (end_index + 1)]
        converted_text = self.tokenizer.convert_ids_to_tokens(converted_text)
        converted_text = "".join(converted_text)

        expected_keys = [
            "input_ids",
            "token_type_ids",
            "position_ids",
            "attention_mask",
            "start_positions",
            "end_positions",
        ]

        assert list(convert_result.keys()) == expected_keys
        assert converted_text == expected_text

    def test_success_convert_when_normal_case(self):
        # Given
        dummy_data = {
            "content": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償110元，民國50年三月110元",
            "result_list": [{"text": "110元", "start": 65, "end": 69}],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_path=self.write_path, write_name=self.write_name)
        output_data = next(read_data_by_chunk(data_path=os.path.join(self.write_path, self.write_name), max_seq_len=512))

        # When
        convert_result = convert_to_uie_format(output_data, self.tokenizer)

        # Then
        self.basic_test(convert_result=convert_result, expected_text=dummy_data["result_list"][0]["text"])

    def test_success_convert_when_cross_chunk(self):
        # Given
        dummy_data = {
            "content": "臺灣苗栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被告黃晨峯上列被告因過失傷害案件，法院判定賠償999元，民國50年三月110元栗地方法院民事裁定110年度苗簡字第563號原告何婷婷被民事裁"
            "定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原民事裁定110年度苗簡字第563號原",
            "result_list": [{"text": "110元", "start": 65, "end": 69}],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_path=self.write_path, write_name=self.write_name)
        output_generator = read_data_by_chunk(data_path=os.path.join(self.write_path, self.write_name), max_seq_len=75)
        _ = next(output_generator)
        second_chunk = next(output_generator)

        # When
        convert_result = convert_to_uie_format(second_chunk, self.tokenizer, 75)

        # Then
        self.basic_test(convert_result=convert_result, expected_text=dummy_data["result_list"][0]["text"])

    def test_success_convert_when_single_word(self):
        # Given
        dummy_data = {
            "content": "臺",
            "result_list": [{"text": "臺", "start": 0, "end": 1}],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_path=self.write_path, write_name=self.write_name)
        output_data = next(read_data_by_chunk(data_path=os.path.join(self.write_path, self.write_name), max_seq_len=512))

        # When
        convert_result = convert_to_uie_format(output_data, self.tokenizer, 512)

        # Then
        self.basic_test(convert_result=convert_result, expected_text=dummy_data["result_list"][0]["text"])

    def test_success_convert_when_empty_result(self):
        # Given
        dummy_data = {
            "content": "臺灣苗栗地方法院",
            "result_list": [],
            "prompt": "醫療費用",
        }
        write_dummy_data_for_model_input(dummy_data=dummy_data, write_path=self.write_path, write_name=self.write_name)
        output_data = next(read_data_by_chunk(data_path=os.path.join(self.write_path, self.write_name), max_seq_len=512))

        # When
        convert_result = convert_to_uie_format(output_data, self.tokenizer, 512)

        # Then
        assert np.where(np.array(convert_result["start_positions"]) == 1.0)[0].tolist() == []
        assert np.where(np.array(convert_result["end_positions"]) == 1.0)[0].tolist() == []

    def teardown_method(self):
        if os.path.exists(self.write_path):
            shutil.rmtree(self.write_path)
