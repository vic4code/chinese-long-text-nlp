import json
import os
import shutil

import pandas as pd
import pytest

from src.judge.merge import do_merge


@pytest.fixture
def merge_config():
    return {
        "information_extraction_path": "./tests/data/",
        "information_extraction_file": "uie_inference_results.csv",
        "text_classification_path": "./tests/data/",
        "text_classification_file": "utc_inference_results.json",
        "save_path": "./tests/data/merge_results",
        "save_file": "merged_results.csv",
    }


def test_success_when_normal_case(merge_config):
    # Given

    # When
    do_merge(merge_config)

    # Then
    assert len(os.listdir(merge_config["save_path"])) == 1


def test_success_when_only_one_result(merge_config):
    # Given
    uie = os.path.join(merge_config["information_extraction_path"], merge_config["information_extraction_file"])
    utc = os.path.join(merge_config["text_classification_path"], merge_config["text_classification_file"])
    with open(utc, "r", encoding="utf8") as f:
        utc_single = json.loads(f.read())
        utc_single = utc_single[0]
    with open("./tests/data/utc_single.json", "w", encoding="utf8") as f:
        out = json.dumps(utc_single, ensure_ascii=False)
        f.write(out)
    uie_single = pd.read_csv(uie)
    uie_single = pd.DataFrame([uie_single.iloc[0, :]])
    uie_single.to_csv("./tests/data/uie_single.csv", header=True, index=False, encoding="utf_8_sig")

    # When
    merge_config["information_extraction_path"] = "./tests/data/"
    merge_config["information_extraction_file"] = "uie_single.csv"
    merge_config["text_classification_path"] = "./tests/data/"
    merge_config["text_classification_file"] = "utc_single.json"
    do_merge(merge_config)

    # Then
    assert len(os.listdir(merge_config["save_path"])) == 1


def test_success_when_uie_result_wrong_name_then_autofind_only_one_csv_file(merge_config):
    # Given
    merge_config["information_extraction_file"] = "WRONG_NAME.txt"
    merge_config["text_classification_file"] = "ANOTHER_WRONG_NAME.json"

    # When
    do_merge(merge_config)

    # Then
    assert len(os.listdir(merge_config["save_path"])) == 1


def test_success_when_save_name_with_txt_then_automodify_csv(merge_config):
    # Given
    merge_config["save_file"] = "test_save_name.txt"

    # When
    do_merge(merge_config)

    # Then
    result = os.listdir(merge_config["save_path"])
    expected_name_start = "test_save_name"
    expected_extension = ".csv"
    assert len(result) == 1
    name, extension = os.path.splitext(result[0])
    assert name[:14] == expected_name_start
    assert expected_extension == extension


def test_fail_when_uie_result_wrong_name_then_find_multiple_csv_file(merge_config):
    # Given
    with open("./tests/data/uie_single.csv", "w", encoding="utf8") as f:
        f.write("")
    merge_config["information_extraction_file"] = "WRONG_NAME.txt"

    # When
    with pytest.raises(ValueError) as error:
        do_merge(merge_config)

    # Then
    expected_error = (
        "Cannot find the file: ./tests/data/WRONG_NAME.txt. " "Please adjust merge_results_config.yaml with correct file path and name."
    )
    assert str(error.value) == expected_error


def test_fail_when_data_diff_then_raise_error(merge_config):
    # Given
    uie = os.path.join(merge_config["information_extraction_path"], merge_config["information_extraction_file"])
    utc = os.path.join(merge_config["text_classification_path"], merge_config["text_classification_file"])
    with open(utc, "r", encoding="utf8") as f:
        utc_single = json.loads(f.read())
        utc_single = utc_single[0]
    with open("./tests/data/utc_single.json", "w", encoding="utf8") as f:
        out = json.dumps(utc_single, ensure_ascii=False)
        f.write(out)
    uie_single = pd.read_csv(uie)
    uie_single.to_csv("./tests/data/uie_single.csv", header=True, index=False, encoding="utf_8_sig")

    # When
    merge_config["information_extraction_path"] = "./tests/data/"
    merge_config["information_extraction_file"] = "uie_single.csv"
    merge_config["text_classification_path"] = "./tests/data/"
    merge_config["text_classification_file"] = "utc_single.json"
    with pytest.raises(ValueError) as error:
        do_merge(merge_config)

    # Then
    expected_error = "Length of uie result is not equal to length of utc result: 3 != 1." " Please check if the inference data is aligned."
    assert str(error.value) == expected_error


def test_fail_when_data_value_diff_then_raise_error(merge_config):
    # Given
    uie = os.path.join(merge_config["information_extraction_path"], merge_config["information_extraction_file"])
    utc = os.path.join(merge_config["text_classification_path"], merge_config["text_classification_file"])
    with open(utc, "r", encoding="utf8") as f:
        utc_single = json.loads(f.read())
        utc_single[0]["jyear"] = "WRONG_YEAR"
    with open("./tests/data/utc_single.json", "w", encoding="utf8") as f:
        out = json.dumps(utc_single, ensure_ascii=False)
        f.write(out)
    uie_single = pd.read_csv(uie)
    uie_single.to_csv("./tests/data/uie_single.csv", header=True, index=False, encoding="utf_8_sig")

    # When
    merge_config["information_extraction_path"] = "./tests/data/"
    merge_config["information_extraction_file"] = "uie_single.csv"
    merge_config["text_classification_path"] = "./tests/data/"
    merge_config["text_classification_file"] = "utc_single.json"
    with pytest.raises(ValueError) as error:
        do_merge(merge_config)

    # Then
    expected_error = (
        "The length of data has been changed after using pd.merge. Before: 3 != After: 2. "
        "Please check if the data of uie result and utc result are aligned. All values must be the same if on the same column names."
    )
    assert str(error.value) == expected_error


def teardown_function():
    # Delete eval output
    if os.path.exists("./tests/data/merge_results"):
        shutil.rmtree("./tests/data/merge_results")

    if os.path.isfile("./tests/data/utc_single.json"):
        os.remove("./tests/data/utc_single.json")
        os.remove("./tests/data/uie_single.csv")
