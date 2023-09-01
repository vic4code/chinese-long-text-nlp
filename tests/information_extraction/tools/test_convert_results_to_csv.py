import json
import os

import pandas as pd

from src.judge.information_extraction.tools.convert_results_to_csv import (
    COLUMN_NAME_OF_JSON_CONTENT,
    adjust_verdict_to_csv_format,
    uie_result_fill_null_entity,
    uie_result_key_remain,
    uie_result_max_select,
    write_json_list_to_csv,
)


def test_uie_result_fill_null_entity_when_normal_case(uie_dummy_result):
    # Give

    # When
    result = uie_result_fill_null_entity(uie_dummy_result)

    # Then
    expected_result = uie_dummy_result[:]
    expected_result[0].update({"醫療費用": [{"text": "nan", "start": -1, "end": -1, "probability": 0.0}]})
    assert result == expected_result


def test_uie_result_max_select_when_normal_case(uie_dummy_result):
    # Give
    uie_dummy_result[0]["精神慰撫金額"][0]["probability"] = 0.9  # make the same prob. then get the first
    uie_dummy_result[0]["精神慰撫金額"][1]["probability"] = 0.9
    uie_dummy_result[0]["薪資收入"].append({"text": "999", "start": 123, "end": 456, "probability": 0.99})
    uie_dummy_result[0].update({"醫療費用": [{"text": "nan", "start": -1, "end": -1, "probability": 0.0}]})

    # When
    result = uie_result_max_select(uie_dummy_result)

    # Then
    expected_result = [
        {
            "精神慰撫金額": [
                {"text": "150,000元", "start": 5188, "end": 5196, "probability": 0.9},
            ],
            "薪資收入": [
                {"text": "999", "start": 123, "end": 456, "probability": 0.99},
            ],
            "醫療費用": [{"text": "nan", "start": -1, "end": -1, "probability": 0.0}],
        }
    ]
    assert result == expected_result


def test_uie_result_key_remain_when_single_key(uie_dummy_result):
    # Give
    uie_dummy_result[0].update({"醫療費用": [{"text": "nan", "start": -1, "end": -1, "probability": 0.0}]})

    # When
    result = uie_result_key_remain(uie_dummy_result, remain_key_in_csv=["start"])

    # Then
    expected_result = [
        {
            "精神慰撫金額": [{"start": 5188}, {"start": 5507}],
            "薪資收入": [{"start": 4452}],
            "醫療費用": [{"start": -1}],
        }
    ]
    assert result == expected_result


def test_uie_result_key_remain_when_full_key(uie_dummy_result):
    # Give
    uie_dummy_result[0].update({"醫療費用": [{"text": "nan", "start": -1, "end": -1, "probability": 0.0}]})

    # When
    result = uie_result_key_remain(uie_dummy_result, remain_key_in_csv=["start", "text", "probability", "end"])

    # Then
    expected_result = [
        {
            "精神慰撫金額": [
                {"start": 5188, "text": "150,000元", "probability": 0.9977151884524105, "end": 5196},
                {"start": 5507, "text": "150,000", "probability": 0.9481814888050799, "end": 5514},
            ],
            "薪資收入": [
                {"start": 4452, "text": "80,856", "probability": 0.9853195936072368, "end": 4458},
            ],
            "醫療費用": [{"start": -1, "text": "nan", "probability": 0.0, "end": -1}],
        }
    ]
    assert result == expected_result


def test_adjust_verdict_to_csv_format_when_normal_case(uie_dummy_result):
    # Given
    with open("./tests/information_extraction/data/inference_results/inference_results_example.json", "r", encoding="utf8") as f:
        result = json.loads(f.read())
        result = [result[0]]
    result[0]["InferenceResults"] = uie_result_fill_null_entity(result[0]["InferenceResults"])
    result[0]["InferenceResults"] = uie_result_max_select(result[0]["InferenceResults"])
    result[0]["InferenceResults"] = uie_result_key_remain(result[0]["InferenceResults"], remain_key_in_csv=["text"])

    # When
    result[0] = adjust_verdict_to_csv_format(verdict=result[0], remain_key_in_csv=["text"])

    # Then
    assert "精神慰撫金額" in list(result[0].keys())
    assert "醫療費用" in list(result[0].keys())
    assert "薪資收入" in list(result[0].keys())


def test_adjust_verdict_to_csv_format_when_no_drop_out():
    # Given
    with open("./tests/information_extraction/data/inference_results/inference_results_example.json", "r", encoding="utf8") as f:
        result = json.loads(f.read())
        result = [result[0]]
    result[0]["InferenceResults"] = uie_result_fill_null_entity(result[0]["InferenceResults"])
    result[0]["InferenceResults"] = uie_result_max_select(result[0]["InferenceResults"])
    result[0]["InferenceResults"] = uie_result_key_remain(result[0]["InferenceResults"], remain_key_in_csv=["text"])

    # When
    result[0] = adjust_verdict_to_csv_format(verdict=result[0], remain_key_in_csv=["text"], drop_keys=[])

    # Then
    assert COLUMN_NAME_OF_JSON_CONTENT in list(result[0].keys())
    assert "InferenceResults" in list(result[0].keys())


def test_adjust_verdict_to_csv_format_when_add_drop_key():
    # Given
    with open("./tests/information_extraction/data/inference_results/inference_results_example.json", "r", encoding="utf8") as f:
        result = json.loads(f.read())
        result = [result[0]]
    result[0]["InferenceResults"] = uie_result_fill_null_entity(result[0]["InferenceResults"])
    result[0]["InferenceResults"] = uie_result_max_select(result[0]["InferenceResults"])
    result[0]["InferenceResults"] = uie_result_key_remain(result[0]["InferenceResults"], remain_key_in_csv=["text"])

    # When
    result[0] = adjust_verdict_to_csv_format(
        verdict=result[0],
        remain_key_in_csv=["text"],
        drop_keys=[COLUMN_NAME_OF_JSON_CONTENT, "InferenceResults", "jyear"],
    )

    # Then
    assert COLUMN_NAME_OF_JSON_CONTENT not in list(result[0].keys())
    assert "InferenceResults" not in list(result[0].keys())
    assert "jyear" not in list(result[0].keys())


def test_write_json_list_to_csv_when_write_paritial_keys():
    # Given
    write_keys = ["jyear", "jid", "精神慰撫金額"]
    with open("./tests/information_extraction/data/inference_results/inference_results_example.json", "r", encoding="utf8") as f:
        input = json.loads(f.read())
        input = [input[0]]
    input[0]["InferenceResults"] = uie_result_fill_null_entity(input[0]["InferenceResults"])
    input[0]["InferenceResults"] = uie_result_max_select(input[0]["InferenceResults"])
    input[0]["InferenceResults"] = uie_result_key_remain(input[0]["InferenceResults"], remain_key_in_csv=["text"])
    input[0] = adjust_verdict_to_csv_format(verdict=input[0], remain_key_in_csv=["text"])

    # When
    write_json_list_to_csv(input, write_keys=write_keys, save_path="./tests/information_extraction/data/inference_results/test.csv")

    # Then
    result = pd.read_csv("./tests/information_extraction/data/inference_results/test.csv")
    assert list(result.columns) == write_keys
    os.remove("./tests/information_extraction/data/inference_results/test.csv")
