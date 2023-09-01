import json
import os

from src.judge.information_extraction import COLUMN_NAME_OF_JSON_CONTENT
from src.judge.information_extraction.tools.convert_results_to_labelstudio import (
    convert_to_labelstudio,
    flatten_uie_output,
    read_uie_inference_results,
)

DATA_PATH = "./tests/information_extraction/data/inference_results/"
DATA_NAME = "inference_results_example.json"


def test_convert_to_labelstudio_when_normal_case():
    # Given
    uie_result_list = read_uie_inference_results(os.path.join(DATA_PATH, DATA_NAME))
    mail = "test@gmail.com"
    save_path = DATA_PATH
    save_name = "test_result.json"

    # When
    convert_to_labelstudio(uie_result_list, mail, save_path, save_name)
    with open(os.path.join(save_path, save_name), "r", encoding="utf8") as f:
        result = json.loads(f.read())

    # Then
    expected_length = 3
    expected_key = "annotations"
    expected_key_in_data = COLUMN_NAME_OF_JSON_CONTENT
    assert len(result) == expected_length
    assert expected_key in list(result[0].keys())
    assert expected_key_in_data in list(result[0]["data"].keys())
    os.remove(os.path.join(save_path, save_name))


def test_read_uie_inference_results_when_normal_case():
    # Given

    # When
    result = read_uie_inference_results(os.path.join(DATA_PATH, DATA_NAME))

    # Then
    expected_length = 3
    expected_key = COLUMN_NAME_OF_JSON_CONTENT
    assert len(result) == expected_length
    assert expected_key in list(result[0].keys())


def test_read_uie_inference_results_when_single_case():
    # Given
    tmp = read_uie_inference_results(os.path.join(DATA_PATH, DATA_NAME))
    tmp_dir = os.path.join(DATA_PATH, "single_result.json")
    with open(tmp_dir, "w", encoding="utf8") as f:
        out = json.dumps(tmp[0], ensure_ascii=False)
        f.write(out)

    # When
    result = read_uie_inference_results(tmp_dir)

    # Then
    expected_length = 1
    expected_key = COLUMN_NAME_OF_JSON_CONTENT
    assert len(result) == expected_length
    assert expected_key in list(result[0].keys())
    os.remove(tmp_dir)


def test_flatten_uie_output_when_output_null():
    # Given
    uie_dummy_result = [{}]

    # When
    result = flatten_uie_output(uie_dummy_result)

    # Then
    expected_result = []
    assert result == expected_result


def test_convert_to_labelstudio_when_only_single_result():
    uie_dummy_result = [
        {
            "精神慰撫金額": [
                {"text": "150,000元", "start": 5188, "end": 5196, "probability": 0.9977151884524105},
                {"text": "150,000", "start": 5507, "end": 5514, "probability": 0.9481814888050799},
            ],
            "薪資收入": [
                {"text": "80,856", "start": 4452, "end": 4458, "probability": 0.9853195936072368},
            ],
        }
    ]

    # When
    result = flatten_uie_output(uie_dummy_result)

    # Then
    expected_length = 3
    expected_0 = {"text": "150,000元", "start": 5188, "end": 5196, "probability": 0.9977151884524105, "entity": "精神慰撫金額"}
    assert len(result) == expected_length
    assert result[0] == expected_0
