# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import PromptModelForSequenceClassification, PromptTrainer, PromptTuningArguments, UTCTemplate
from paddlenlp.transformers import UTC, AutoTokenizer, PretrainedTokenizer
from paddlenlp.utils.log import logger

from .utils import UTCLoss, convert_results_to_label_studio, read_inference_dataset


@dataclass
class DataArguments:
    verdict_sheet_path: str = field(
        default="./data/inference/infer_example.json", metadata={"help": "Original text data path for model inference."}
    )
    data_file_to_inference: str = field(default="./data/inference/infer_example.json", metadata={"help": "Data path for model inference."})
    threshold: float = field(default=0.5, metadata={"help": "The threshold to produce predictions."})
    label_file: str = field(
        default="./data/modeling/text_classification/label_studio/labels.txt", metadata={"help": "Labels file for classification task."}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="utc-base", metadata={"help": "The utc model name."})
    checkpoint_path: str = field(
        default="./models/text_classification/utc/checkpoints/model_best",
        metadata={"help": "Finetuned model checkpoint path to be loaded."},
    )


class InferenceUTCTemplate(UTCTemplate):
    template_special_tokens = ["text", "hard", "sep", "cls", "options"]

    def __init__(self, tokenizer: PretrainedTokenizer, max_length: int, prompt: str = None):
        prompt = (
            (
                "{'options': 'choices', 'add_omask': True, 'position': 0, 'token_type': 1}"
                "{'sep': None, 'token_type': 0, 'position': 0}{'text': 'text_a'}{'sep': None, 'token_type': 1}{'text': 'text_b'}"
            )
            if prompt is None
            else prompt
        )
        super(UTCTemplate, self).__init__(prompt, tokenizer, max_length)
        self.max_position_id = self.tokenizer.model_max_length - 1
        self.max_length = max_length
        if not self._has_options():
            raise ValueError("Expected `options` and `add_omask` are in defined prompt, but got {}".format(self.prompt))

    def _has_options(self):
        for part in self.prompt:
            if "options" in part and "add_omask" in part:
                return True
        return False

    def build_inputs_with_prompt(self, example: Dict[str, Any], prompt: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        inputs = super(UTCTemplate, self).build_inputs_with_prompt(example, prompt)
        for index, part in enumerate(inputs):
            if "cls" in part:
                inputs[index] = self.tokenizer.cls_token

        return inputs

    def encode(self, example: Dict[str, Any], use_mask: bool = False):
        input_dict = super(UTCTemplate, self).encode(example)

        # Set OMASK and MASK positions and labels for options.
        omask_token_id = self.tokenizer.convert_tokens_to_ids("[O-MASK]")
        input_dict["omask_positions"] = np.where(np.array(input_dict["input_ids"]) == omask_token_id)[0].squeeze().tolist()

        sep_positions = np.where(np.array(input_dict["input_ids"]) == self.tokenizer.sep_token_id)[0].squeeze().tolist()
        input_dict["cls_positions"] = sep_positions[0]

        # Limit the maximum position ids.
        position_ids = np.array(input_dict["position_ids"])
        position_ids[position_ids > self.max_position_id] = self.max_position_id
        input_dict["position_ids"] = position_ids.tolist()

        return input_dict

    def create_prompt_parameters(self):
        return None

    def process_batch(self, input_dict):
        return input_dict


def inference(model_args, data_args, infer_args):
    paddle.set_device(infer_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = UTC.from_pretrained(model_args.model_name_or_path)

    # Define template for preprocess and verbalizer for postprocess.
    template = InferenceUTCTemplate(tokenizer, infer_args.max_seq_length)

    # Load and preprocess dataset.
    try:
        with open(data_args.label_file, "r", encoding="utf-8") as fp:
            choices = [x.strip() for x in fp]
    except Exception:
        raise ValueError("Invalid label_file. Please use file with one label per line or set `label_file` with condidate labels.")

    try:
        with open(data_args.data_file_to_inference, "r", encoding="utf-8") as fp:
            test_data = json.load(fp)
    except Exception:
        raise ValueError(
            "Invalid file or path. The input json file must be exported from CA verdict database and with 'jfull_compress' included."
        )

    test_ds = load_dataset(read_inference_dataset, data_path=data_args.data_file_to_inference, lazy=False, label_file=data_args.label_file)

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=infer_args.freeze_plm, freeze_dropout=infer_args.freeze_dropout
    )
    if model_args.checkpoint_path is not None:
        model_state = paddle.load(os.path.join(model_args.checkpoint_path, "model_state.pdparams"))
        prompt_model.set_state_dict(model_state)

    # Define the metric function.
    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=infer_args,
        criterion=UTCLoss(),
        train_dataset=None,
        eval_dataset=None,
        callbacks=None,
    )

    # Inference with utc model
    test_ret = trainer.predict(test_ds)

    if not os.path.exists(infer_args.output_dir):
        os.makedirs(infer_args.output_dir)

    with open(os.path.join(infer_args.output_dir, "inference_results.json"), "w", encoding="utf-8") as fp:
        preds = paddle.nn.functional.sigmoid(paddle.to_tensor(test_ret.predictions))

        logger.info("Start to inference data with utc...")

        for index, example in enumerate(test_data):
            if "jfull_compress" not in example and not isinstance(example["jfull_compress"], str):
                raise ValueError("The inference file must include 'jfull_compress' key from verdict sheet database.")

            del example["jfull_compress"]
            pred_ids = paddle.where(preds[index] > data_args.threshold)[0].tolist()

            example["pred_labels"] = {}
            for choice in choices:
                example["pred_labels"][choice] = 0

            if pred_ids:
                for pred_id in pred_ids:
                    example["pred_labels"][choices[pred_id[0]]] = 1

        json.dump(test_data, fp, ensure_ascii=False, indent=4)

    convert_results_to_label_studio(
        verdict_sheet_path=data_args.verdict_sheet_path,
        inference_results_path=os.path.join(infer_args.output_dir, "inference_results.json"),
        output_path=infer_args.output_dir,
        filename="inference_results_label_studio.json",
    )


def main(args: dict):
    # Parse the arguments.
    model_args, data_args, infer_args = (
        ModelArguments(**args["model_args"]),
        DataArguments(**args["data_args"]),
        PromptTuningArguments(**args["infer_args"]),
    )
    infer_args.print_config(model_args, "Model")
    infer_args.print_config(data_args, "Data")

    # Inference
    inference(model_args, data_args, infer_args)
