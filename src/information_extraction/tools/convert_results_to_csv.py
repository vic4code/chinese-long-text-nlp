import argparse
import csv
import json
import os
from typing import List

from src.judge.information_extraction import (
    COLUMN_NAME_OF_JSON_CONTENT,
    ENTITY_TYPE,
    KEYS_MAPPING_TO_CSV_TABLE,
    REMAIN_KEYS,
    logger,
)


def read_uie_inference_results(path: str) -> List[dict]:
    """Get the UIE results made by run_infer.py

    Args:
        path (str): Path of UIE results.

    Returns:
        _type_: List of UIE results.
    """
    with open(path, "r", encoding="utf8") as f:
        infile = f.read()
        if len(infile) == 0:
            raise ValueError(f"Data not found in file: {path}")
        result_list = json.loads(infile)
    return result_list if isinstance(result_list, list) else [result_list]


# fill nan
def uie_result_fill_null_entity(uie_result, fill_text_when_null: str = "nan"):
    for entity in ENTITY_TYPE:
        if not uie_result[0].get(entity):
            uie_result[0].update({entity: [{"text": fill_text_when_null, "start": -1, "end": -1, "probability": 0.0}]})
    return uie_result


# max filter
def uie_result_max_select(uie_result):
    new_result = [{}]
    for entity in uie_result[0]:
        new_result[0][entity] = [sorted(uie_result[0][entity], key=lambda x: x["probability"], reverse=True)[0]]
    return new_result


# select key
def uie_result_key_remain(uie_result, remain_key_in_csv: List[str]):
    new_result = [{}]
    for entity in uie_result[0]:
        tmp_list = []
        for each_result_in_entity in uie_result[0][entity]:
            tmp_dict = {}
            for key in remain_key_in_csv:
                tmp_dict.update({key: each_result_in_entity[key]})
            tmp_list.append(tmp_dict)
        new_result[0][entity] = tmp_list
    return new_result


# only work in single result
def adjust_verdict_to_csv_format(
    verdict,
    remain_key_in_csv: List[str],
    drop_keys: List[str] = [COLUMN_NAME_OF_JSON_CONTENT, "InferenceResults"],
):
    update_entity_result = {}
    for entity in verdict["InferenceResults"][0]:
        for key in remain_key_in_csv:
            if key == "text":
                update_entity_result.update({entity: verdict["InferenceResults"][0][entity][0]["text"]})
            else:
                update_entity_result.update(
                    {f"{KEYS_MAPPING_TO_CSV_TABLE[key]}_for_{entity}": verdict["InferenceResults"][0][entity][0][key]}
                )

    for drop_key in drop_keys:
        verdict.pop(drop_key)

    verdict.update(update_entity_result)
    return verdict


def write_json_list_to_csv(file_list, write_keys=None, save_path="./uie_result_for_csv.csv"):
    header = write_keys if write_keys else list(file_list[0].keys())
    with open(save_path, "w", encoding="utf_8_sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for file in file_list:
            data = [file[key] for key in header]
            writer.writerow(data)


def convert_to_csv(uie_inference_results, save_path, save_name):
    for i, inference_result in enumerate(uie_inference_results):
        uie_inference_results[i]["InferenceResults"] = uie_result_fill_null_entity(uie_result=inference_result["InferenceResults"])
        uie_inference_results[i]["InferenceResults"] = uie_result_max_select(uie_result=inference_result["InferenceResults"])
        uie_inference_results[i]["InferenceResults"] = uie_result_key_remain(
            uie_result=inference_result["InferenceResults"], remain_key_in_csv=REMAIN_KEYS
        )
        uie_inference_results[i] = adjust_verdict_to_csv_format(inference_result, remain_key_in_csv=REMAIN_KEYS)

    write_json_list_to_csv(uie_inference_results, save_path=os.path.join(save_path, save_name))


if __name__ == "__main__":
    """將「run_infer.py」inference 產生的結果，轉換成 csv 格式。

    Example:
        python ./src/judge/information_extraction/tools/convert_results_to_csv.py \
            --uie_results_path ./reports/information_extraction/inference_results/inference_results.json

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--uie_results_path", type=str)
    parser.add_argument("--save_path", type=str, default="./")
    parser.add_argument("--save_name", type=str, default="uie_result_for_csv.csv")
    args = parser.parse_args()

    logger.info(f"Read the uie results from {args.uie_results_path}")
    logger.info("Start converting the results into csv...")

    uie_inference_results = read_uie_inference_results(path=args.uie_results_path)

    convert_to_csv(uie_inference_results, args.save_path, args.save_name)

    logger.info(f"Write the csv into {os.path.join(args.save_path, args.save_name)}")
    logger.info("Finish.")
