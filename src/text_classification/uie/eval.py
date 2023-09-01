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
from functools import partial

import paddle
from paddlenlp.data import DataCollatorWithPadding
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import SpanEvaluator
from paddlenlp.transformers import UIE, UIEM, AutoTokenizer
from paddlenlp.utils.log import logger

from src.judge.information_extraction.utils.data_utils import convert_to_uie_format, read_data_by_chunk

from .utils import create_data_loader


@dataclass
class EvalArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    checkpoint_path: str = field(
        default="./models/text_classification/uie/checkpoints/model_best",
        metadata={"help": "The path of the model checkpoint to evaluate."},
    )
    test_path: str = field(
        default="./data/modeling/text_classification/model_train/uie/test.txt", metadata={"help": "The path of the test dataset."}
    )
    output_dir: str = field(
        default="./models/text_classification/uie/checkpoints/model_best/test_results",
        metadata={"help": "The path of the test results to save."},
    )
    batch_size: int = field(default=16, metadata={'description': 'Batch size to eval the test data.'})
    device: str = field(default="gpu", metadata={"help": "The device to run model."})
    max_seq_length: int = field(default=768, metadata={'description': 'Max sequence length for model inputs.'})
    multilingual: bool = field(default=False, metadata={'description': 'If multilingual or not to load the pretrained model.'})
    schema_lang: str = field(default="ch", metadata={'description': 'Schema language.'})


@paddle.no_grad()
def compute_metrics(model, metric, data_loader, multilingual=False):
    """
    Given a dataset, it evals model and computes the metric.
    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        multilingual(bool): Whether is the multilingual model.
    """
    model.eval()
    metric.reset()
    for batch in data_loader:
        if multilingual:
            start_prob, end_prob = model(batch["input_ids"], batch["position_ids"])
        else:
            start_prob, end_prob = model(batch["input_ids"], batch["token_type_ids"], batch["position_ids"], batch["attention_mask"])

        start_ids = paddle.cast(batch["start_positions"], "float32")
        end_ids = paddle.cast(batch["end_positions"], "float32")
        num_correct, num_infer, num_label = metric.compute(start_prob, end_prob, start_ids, end_ids)
        metric.update(num_correct, num_infer, num_label)
    precision, recall, f1 = metric.accumulate()
    model.train()
    return precision, recall, f1


def evaluate(args):
    paddle.set_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    if args.multilingual:
        model = UIEM.from_pretrained(args.checkpoint_path)
    else:
        model = UIE.from_pretrained(args.checkpoint_path)

    test_ds = load_dataset(
        read_data_by_chunk,
        data_path=args.test_path,
        max_seq_len=args.max_seq_length,
        lazy=False,
    )

    trans_fn = partial(
        convert_to_uie_format,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_length,
    )

    class_dict = {}
    class_dict["all_classes"] = test_ds

    for key in class_dict.keys():
        test_ds = class_dict[key]
        test_ds = test_ds.map(trans_fn)

        data_collator = DataCollatorWithPadding(tokenizer)

        test_data_loader = create_data_loader(test_ds, mode="test", batch_size=args.batch_size, trans_fn=data_collator)

        metric = SpanEvaluator()
        precision, recall, f1 = compute_metrics(model, metric, test_data_loader, args.multilingual)
        logger.info("-----------------------------")
        logger.info("Class Name: %s" % key)
        logger.info("Evaluation Precision: %.5f | Recall: %.5f | F1: %.5f" % (precision, recall, f1))

        metrics = {"precision": precision, "recall": recall, "f1": f1}

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        with open(os.path.join(args.output_dir, "test_metrics.json"), "w", encoding="utf-8") as fp:
            json.dump(metrics, fp)


def main(args: dict):
    eval_args = EvalArgs(**args)
    evaluate(eval_args)
