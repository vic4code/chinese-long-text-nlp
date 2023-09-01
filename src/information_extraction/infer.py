import json
import os
import re
from datetime import datetime
from typing import Callable, List

from paddlenlp import Taskflow
from tqdm import tqdm

from . import COLUMN_NAME_OF_JSON_CONTENT, ENTITY_TYPE, logger
from .tools.convert_csv_money import convert_money
from .tools.convert_results_to_csv import convert_to_csv
from .tools.convert_results_to_labelstudio import convert_to_labelstudio


class Processer:
    def __init__(
        self,
        select_strategy: str = "all",
        threshold: float = 0.5,
        select_key: List[str] = ["text", "start", "end", "probability"],
        is_regularize_data: bool = False,
    ) -> None:
        self.select_strategy_fun = eval("self._" + select_strategy + "_postprocess")
        self.threshold = threshold if threshold else 0.5
        self.select_key = select_key if select_key else ["text", "start", "end", "probability"]
        self.is_regularize_data = is_regularize_data
        self.data = None

    def _key_filter(strategy_fun):
        def select_key(self, each_entity_results):
            each_entity_results = strategy_fun(self, each_entity_results)
            for i, each_entity_result in enumerate(each_entity_results):
                each_entity_results[i] = {key: each_entity_result[key] for key in self.select_key}
            return each_entity_results

        return select_key

    def preprocess(self, text):
        """
        Override this method if you want to inject some custom behavior
        """

        if self.is_regularize_data:
            for re_term in [r"\n", r" ", r"\u3000", r"\\n"]:
                text = re.sub(re_term, "", text)
        return text

    def postprocess(self, results):
        """
        Override this method if you want to inject some custom behavior

        results: [
            {'精神慰撫金額': [{'text': '30,000元', 'start': 2659, 'end': 2666, 'probability': 0.992}],
             '醫療費用': [{'text': '4,373元', 'start': 2714, 'end': 2720, 'probability': 0.928}]}]
        """
        new_result = [{}]
        for entity in results[0]:
            tmp = self.select_strategy_fun(results[0][entity])
            new_result[0].update({entity: tmp})
        return new_result

    @_key_filter
    def _max_postprocess(self, each_entity_results):
        return [sorted(each_entity_results, key=lambda x: x["probability"], reverse=True)[0]]

    @_key_filter
    def _threshold_postprocess(self, each_entity_results):
        return list(filter(lambda x: x["probability"] > self.threshold, each_entity_results))

    @_key_filter
    def _all_postprocess(self, each_entity_results):
        return each_entity_results

    @_key_filter
    def _CustomizeYourName_postprocess(self, each_entity_results):
        """
        1. Set --select_strategy CustomizeYourName
           Any select strategy can be implemented here.

        2. each_entity_results (example):
            [{'text': '22,154元', 'start': 1487, 'end': 1494, 'probability': 0.46},
             {'text': '2,954元', 'start': 3564, 'end': 3570, 'probability': 0.80}]
        """
        pass


def inference(
    data_file: str,
    schema: List[str],
    device_id: int = 0,
    text_list: List[str] = None,
    precision: str = "fp32",
    batch_size: int = 1,
    model: str = "uie-base",
    task_path: str = None,
    postprocess_fun: Callable = lambda x: x,
    preprocess_fun: Callable = lambda x: x,
):
    if not os.path.exists(data_file) and not text_list:
        raise ValueError(f"Data not found in {data_file}. Please input the correct path of data.")

    if task_path:
        if not os.path.exists(task_path):
            raise ValueError(f"{task_path} is not a directory.")
        uie = Taskflow(
            "information_extraction",
            schema=schema,
            task_path=task_path,
            precision=precision,
            batch_size=batch_size,
            device_id=device_id,
        )
    else:
        uie = Taskflow(
            "information_extraction",
            schema=schema,
            model=model,
            precision=precision,
            batch_size=batch_size,
            device_id=device_id,
        )

    final_results = []
    if not text_list:
        _, extension = os.path.splitext(data_file)
        if extension == ".txt":
            with open(data_file, "r", encoding="utf8") as f:
                final_results = [{COLUMN_NAME_OF_JSON_CONTENT: line.strip()} for line in f]
        elif extension == ".json":
            with open(data_file, "r", encoding="utf8") as f:
                final_results = json.loads(f.read())
        else:
            raise ValueError(f"Invalid file input: {extension}. Please use .txt or .json instead.")
    else:
        final_results = [{COLUMN_NAME_OF_JSON_CONTENT: line} for line in text_list]

    for i in tqdm(range(len(final_results))):
        final_results[i].update({COLUMN_NAME_OF_JSON_CONTENT: preprocess_fun(final_results[i][COLUMN_NAME_OF_JSON_CONTENT])})
        inference_result = postprocess_fun(uie(final_results[i][COLUMN_NAME_OF_JSON_CONTENT]))
        final_results[i].update({"InferenceResults": inference_result})
    return final_results


def main(args: dict):
    data_args, taskflow_args, strategy_args = args["data_args"], args["taskflow_args"], args["strategy_args"]

    uie_processer = Processer(
        select_strategy=strategy_args["select_strategy"],
        threshold=strategy_args["select_strategy_threshold"],
        select_key=strategy_args["select_key"],
        is_regularize_data=data_args["is_regularize_data"],
    )

    logger.info("Start Inference...")

    inference_result = inference(
        data_file=data_args["data_file"],
        device_id=taskflow_args["device_id"],
        schema=ENTITY_TYPE,
        text_list=data_args["text_list"],
        precision=taskflow_args["precision"],
        batch_size=taskflow_args["batch_size"],
        model=taskflow_args["model"],
        task_path=taskflow_args["task_path"],
        postprocess_fun=uie_processer.postprocess,
        preprocess_fun=uie_processer.preprocess,
    )

    logger.info("End Inference...")

    if data_args["save_dir"]:
        now = datetime.now().strftime("%m%d_%I%M%S")
        save_name_sep = data_args["save_name"].split(".")

        if len(save_name_sep) != 2:
            raise ValueError(f"File name error on {data_args['save_dir']}.")

        save_name = save_name_sep[0] + "_" + now + "." + save_name_sep[1]

        if not os.path.exists(data_args["save_dir"]):
            logger.warning(f"{data_args['save_dir']} is not found. Auto-create the dir.")
            os.makedirs(data_args["save_dir"])

        logger.info(f"Write the results into {os.path.join(data_args['save_dir'], save_name)}.")
        with open(os.path.join(data_args["save_dir"], save_name), "w", encoding="utf8") as f:
            jsonString = json.dumps(inference_result, ensure_ascii=False)
            f.write(jsonString)

    if data_args["is_export_labelstudio"]:
        convert_to_labelstudio(
            inference_result,
            "YOUR_MAIL@gamil.com",
            save_path=data_args["save_dir"],
            save_name=save_name_sep[0] + "_" + now + "_labelstudio_type.json",
        )

    if data_args["is_export_csv"]:
        convert_to_csv(inference_result, save_path=data_args["save_dir"], save_name=save_name_sep[0] + "_" + now + "_csv_type.csv")
        if data_args["is_regularize_csv_money"]:
            convert_money(
                csv_results_path=os.path.join(data_args["save_dir"], save_name_sep[0] + "_" + now + "_csv_type.csv"),
                save_path=data_args["save_dir"],
                save_name=save_name_sep[0] + "_" + now + "_regularized_csv_type.csv",
            )

    logger.info("Finish.")
