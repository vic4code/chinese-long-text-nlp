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

import paddle
from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import PromptModelForSequenceClassification, PromptTrainer, PromptTuningArguments, UTCTemplate
from paddlenlp.transformers import UTC, AutoTokenizer

from .metric import MetricReport
from .utils import UTCLoss, read_local_dataset


@dataclass
class DataArguments:
    test_path: str = field(
        default="./data/modeling/text_classification/model_train/utc/dataset/test.txt", metadata={"help": "Test dataset file name."}
    )
    threshold: float = field(default=0.5, metadata={"help": "The threshold to produce predictions."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="utc-base", metadata={"help": "Build-in pretrained model."})
    checkpoint_path: str = field(
        default="./models/text_classification/utc/checkpoints/model_best",
        metadata={"help": "Finetuned model checkpoint path to be loaded."},
    )


def evaluate(model_args, data_args, eval_args):
    paddle.set_device(eval_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = UTC.from_pretrained(model_args.model_name_or_path)

    # Define template for preprocess and verbalizer for postprocess.
    template = UTCTemplate(tokenizer, eval_args.max_seq_length)

    # Load and preprocess dataset.
    if data_args.test_path is not None:
        test_ds = load_dataset(read_local_dataset, data_path=data_args.test_path, lazy=False)

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=eval_args.freeze_plm, freeze_dropout=eval_args.freeze_dropout
    )
    if model_args.checkpoint_path is not None:
        model_state = paddle.load(os.path.join(model_args.checkpoint_path, "model_state.pdparams"))
        prompt_model.set_state_dict(model_state)

    def compute_metrics(eval_preds):
        metric = MetricReport()
        metric.reset()
        labels = paddle.to_tensor(eval_preds.label_ids, dtype="int64")
        preds = paddle.to_tensor(eval_preds.predictions)

        preds = paddle.nn.functional.sigmoid(preds)
        preds = preds > data_args.threshold

        metric.update(preds, labels)
        micro_f1_score, macro_f1_score, accuracy, precision, recall = metric.accumulate()
        metric.reset()

        return {
            "eval_micro_f1": micro_f1_score,
            "eval_macro_f1": macro_f1_score,
            "accuracy_score": accuracy,
            "precision_score": precision,
            "recall_score": recall,
        }

    trainer = PromptTrainer(
        model=prompt_model,
        tokenizer=tokenizer,
        args=eval_args,
        criterion=UTCLoss(),
        train_dataset=None,
        eval_dataset=None,
        callbacks=None,
        compute_metrics=compute_metrics,
    )

    if data_args.test_path is not None:
        test_ret = trainer.predict(test_ds)
        trainer.log_metrics("test", test_ret.metrics)
        with open(os.path.join(eval_args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as fp:
            json.dump(test_ret.metrics, fp)

        with open(os.path.join(eval_args.output_dir, "test_predictions.json"), "w", encoding="utf-8") as fp:
            preds = paddle.nn.functional.sigmoid(paddle.to_tensor(test_ret.predictions))
            for index, pred in enumerate(preds):
                result = {"id": index}
                result["labels"] = paddle.where(pred > data_args.threshold)[0].tolist()
                result["probs"] = pred[pred > data_args.threshold].tolist()
                fp.write(json.dumps(result, ensure_ascii=False) + "\n")


def main(args: dict):
    # Parse the arguments.
    model_args, data_args, eval_args = (
        ModelArguments(**args["model_args"]),
        DataArguments(**args["data_args"]),
        PromptTuningArguments(**args["eval_args"]),
    )
    eval_args.print_config(model_args, "Model")
    eval_args.print_config(data_args, "Data")

    # Eval test data
    evaluate(model_args, data_args, eval_args)
