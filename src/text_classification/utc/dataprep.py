# coding=utf-8
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
import random
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List

import numpy as np
import paddle
from paddlenlp.utils.log import logger


@dataclass
class LabelStudioArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    label_studio_file: str = field(
        default="../data/label_studio/text_classification.json",
        metadata={"help": "The path of the label studio data to be converted."},
    )
    save_dir: str = field(
        default="./data/modeling/text_classification/model_train/utc/dataset",
        metadata={"help": "The path of the label studio data to be converted."},
    )
    splits: List[float] = field(default_factory=list, metadata={'description': 'Splits for train, dev, test'})
    text_separator: str = field(default="\t", metadata={"help": "Text separator."})
    label_file: str = field(
        default="./data/modeling/text_classification/label_studio/labels.txt",
        metadata={'description': 'The paths of labels for classification'},
    )
    is_shuffle: bool = field(default=True, metadata={'description': 'Is shuffle during training.'})
    seed: int = field(default=1000, metadata={'description': 'Random seed to split data.'})


def set_seed(seed):
    paddle.seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class LabelStudioDataConverter(object):
    """
    DataConverter to convert data export from LabelStudio platform
    """

    def __init__(self, label_file, text_separator):
        super().__init__()
        if os.path.isfile(label_file) and label_file.endswith(".txt"):
            with open(label_file, "r", encoding="utf-8") as fp:
                self.labels = [x.strip() for x in fp]
        else:
            raise ValueError("Invalid label_file. Please use file with one label per line or set `label_file` with condidate labels.")
        self.text_separator = text_separator

    def convert_utc_examples(self, raw_examples):
        utc_examples = []
        for example in raw_examples:
            raw_text = example["data"]["jfull_compress"].split(self.text_separator)
            if len(raw_text) < 1:
                continue
            elif len(raw_text) == 1:
                raw_text.append("")
            elif len(raw_text) > 2:
                raw_text = ["".join(raw_text[:-1]), raw_text[-1]]

            label_list = []
            if example["annotations"][0]["result"]:
                for raw_label in example["annotations"][0]["result"][0]["value"]["choices"]:
                    if raw_label not in self.labels:
                        raise ValueError(f"Label `{raw_label}` not found in label candidates `label_file`. Please recheck the data.")
                    label_list.append(np.where(np.array(self.labels) == raw_label)[0].tolist()[0])

            utc_examples.append(
                {
                    "text_a": raw_text[0],
                    "text_b": raw_text[1],
                    "question": "",
                    "choices": self.labels,
                    "labels": label_list,
                }
            )
        return utc_examples


def do_convert(args):
    set_seed(args.seed)

    tic_time = time.time()
    if not os.path.exists(args.label_studio_file):
        raise ValueError("Please input the correct path of label studio file.")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.splits) != 0 and len(args.splits) != 3:
        raise ValueError("Only []/ len(splits)==3 accepted for splits.")

    def _check_sum(splits):
        return Decimal(str(splits[0])) + Decimal(str(splits[1])) + Decimal(str(splits[2])) == Decimal("1")

    if len(args.splits) == 3 and not _check_sum(args.splits):
        raise ValueError("Please set correct splits, sum of elements in splits should be equal to 1.")

    with open(args.label_studio_file, "r", encoding="utf-8") as fp:
        infile = fp.read()
        if len(infile) == 0:
            raise ValueError(f"Data not found in file: {args.label_studio_file}")
        raw_examples = json.loads(infile)

    if args.is_shuffle:
        indexes = np.random.permutation(len(raw_examples))
        raw_examples = [raw_examples[i] for i in indexes]

    i1, i2, _ = args.splits
    p1 = int(len(raw_examples) * i1)
    p2 = int(len(raw_examples) * (i1 + i2))

    data_converter = LabelStudioDataConverter(args.label_file, args.text_separator)

    train_examples = data_converter.convert_utc_examples(raw_examples[:p1])
    dev_examples = data_converter.convert_utc_examples(raw_examples[p1:p2])
    test_examples = data_converter.convert_utc_examples(raw_examples[p2:])

    if not train_examples:
        train_examples = dev_examples if dev_examples else test_examples

    if not dev_examples:
        dev_examples = test_examples

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        logger.info("Save %d examples to %s." % (count, save_path))

    _save_examples(args.save_dir, "train.txt", train_examples)
    _save_examples(args.save_dir, "dev.txt", dev_examples)
    _save_examples(args.save_dir, "test.txt", test_examples)

    logger.info("Finished! It takes %.2f seconds" % (time.time() - tic_time))


def main(args: dict):
    # Label studio data preparationå
    label_studio_args = LabelStudioArgs(**args)
    logger.info("Preparing dataset from label studio data...")

    # Convert label studio data to the format of model inputs, and split data as train, dev, test
    do_convert(label_studio_args)
