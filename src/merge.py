import json
import os
from datetime import datetime

import pandas as pd

from .information_extraction import logger

UTC_RESULT_LABEL_NAME = "pred_labels"


def detect_file_exist(file_path, file_type):
    dir = os.path.dirname(file_path)
    if not os.path.isfile(file_path):
        logger.warning(f"Cannot find the file: {file_path}")
        logger.warning(f"Automatically find the possibly .{file_type} file in: {dir}")
        possibly_file = [file for file in os.listdir(dir) if os.path.splitext(file)[1] == f".{file_type}"]
        if len(possibly_file) == 1:
            logger.warning(f"Using the possibly file {possibly_file[0]} instead. Please check the reference correct.")
            file_path = os.path.join(dir, possibly_file[0])
        else:
            raise ValueError(
                f"Cannot find the file: {file_path}. Please adjust merge_results_config.yaml" " with correct file path and name."
            )
    return file_path


def get_data_path(config):
    info_path = detect_file_exist(
        file_path=os.path.join(config["information_extraction_path"], config["information_extraction_file"]),
        file_type="csv",
    )
    text_path = detect_file_exist(
        file_path=os.path.join(config["text_classification_path"], config["text_classification_file"]),
        file_type="json",
    )

    if os.path.splitext(info_path)[1] != ".csv":
        raise ValueError(
            "Information extraction task only take '.csv' file. "
            " Please use the ./src/judge/information_extraction/tools/convert_results_to_csv.py to convert the uie results to .csv file."
        )

    if os.path.splitext(text_path)[1] != ".json":
        raise ValueError("Text classification task only take '.json' file. Please check if the file is correct.")

    if not os.path.exists(config["save_path"]):
        os.mkdir(config["save_path"])

    name, extension = os.path.splitext(config["save_file"])
    if extension != ".csv":
        logger.warning(f"Cannot support {extension} as output file. Automatically adjust .csv instead.")
        config["save_file"] = name + ".csv"

    return info_path, text_path, config


def do_merge(args: dict):
    """合併 Information Extraction 和 Text Classification 之推論結果。

    - 參數檔案：merge_results_config.yaml
    - 合併邏輯：pd.merge 中的 inner join，key 為「所有」相同的欄位名稱。
    - 合併檔案：
        - Information Extraction: UIE Inference 結果，並透過 convert_results_to_csv.py 轉換成的 csv 檔。
        - Text Classification: UTC Inference 結果。
    - 使用：python -m judge merge_results

    Raises:
        ValueError: UIE 結果和 UTC 結果長度不相同。請確認兩者任務是「推論同一個檔案」，否則無法合併。
        ValueError: 合併結果和來源長度不同。請確認兩者任務是「推論同一個檔案」。
    """
    logger.info("Start merge...")

    info, text, config = get_data_path(args)

    logger.info(f"Loading the information extraction result on {info}")
    logger.info(f"Loading the text classification result on {text}")

    uie_result = pd.read_csv(info)
    with open(text, "r", encoding="utf8") as f:
        utc_result = json.loads(f.read())
        if not isinstance(utc_result, list):
            utc_result = [utc_result]

    if len(uie_result) != len(utc_result):
        raise ValueError(
            f"Length of uie result is not equal to length of utc result: {len(uie_result)} != {len(utc_result)}."
            " Please check if the inference data is aligned."
        )

    for i, _ in enumerate(utc_result):
        utc_result[i].update(utc_result[i][UTC_RESULT_LABEL_NAME])
        utc_result[i].pop(UTC_RESULT_LABEL_NAME)

    logger.info(f"Merging data length: {len(uie_result)}.")
    logger.info("Merging Method: inner join.")

    utc_result = pd.DataFrame(utc_result)
    merge_key = uie_result.columns.intersection(utc_result.columns).tolist()
    for k in merge_key:
        uie_result[k] = uie_result[k].apply(type(utc_result[k][0]))

    logger.info(f"Merging by the same columns: {merge_key}.")

    merged_results = pd.merge(uie_result, utc_result, on=merge_key, how="inner")

    if len(uie_result) != len(merged_results):
        raise ValueError(
            f"The length of data has been changed after using pd.merge. Before: {len(uie_result)} != After: {len(merged_results)}."
            " Please check if the data of uie result and utc result are aligned. "
            "All values must be the same if on the same column names."
        )

    now = datetime.now().strftime("%m%d_%I%M%S")
    save_name_sep = os.path.splitext(config["save_file"])
    save_name = save_name_sep[0] + "_" + now + save_name_sep[1]
    merged_results.to_csv(os.path.join(config["save_path"], save_name), header=True, index=False, encoding="utf_8_sig")

    logger.info(f"Success! Save the merge results in the {os.path.join(config['save_path'], save_name)} directory.")
    logger.info("Finish.")
