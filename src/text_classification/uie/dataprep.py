import json
import os
import re
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import List

import numpy as np
from paddlenlp.utils.log import logger
from tqdm import tqdm

from .utils import convert_cls_examples, convert_ext_examples, set_seed


@dataclass
class LabelStudioArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    label_studio_file: str = field(
        default="./data/modeling/text_classification/label_studio/uie_input_example.json",
        metadata={"help": "The path of the label studio data which needs to be converted into trian.txt, dev.txt, test.txt"},
    )
    save_dir: str = field(
        default="./data/modeling/text_classification/model_train/uie",
        metadata={"help": "The path to save trian.txt, dev.txt, test.txt dataset."},
    )
    negative_ratio: int = field(default=5, metadata={'help': 'negative_ratio'})
    split_ratio: List[float] = field(default_factory=list, metadata={'help': 'split_ratio for train, dev, test'})  # [train, dev, test]
    task_type: str = field(default="ext", metadata={'help': 'Task type of uie.'})
    options: List[str] = field(default_factory=list, metadata={'help': 'Options for uie model in prompt learning.'})
    prompt_prefix: str = field(default="情感倾向", metadata={'help': 'Prompt prefix for uie.'})
    is_shuffle: bool = field(default=True, metadata={'help': 'Is shuffle during training.'})
    seed: int = field(default=1000, metadata={'help': 'Set random seed to split data.'})
    separator: str = field(default="##", metadata={'help': 'Separator signs.'})
    schema_lang: str = field(default="ch", metadata={'description': 'Schema language.'})


def check_spaces(text):
    # Check if the text contains fullwidth sapce.
    if '　' in text:
        logger.warning("The text contains fullwidth sapce that could affect your model performance.")

    # Check if the text contains halfwidth sapce, and is reserved only when the word is English.
    words = text.split()
    for word in words:
        if re.match(r'[A-Za-z]+$', word):
            continue
        if ' ' in word:
            logger.warning("The text contains invalid halfwidth sapce that could affect your model performance.")


def append_attrs(data, item, label_id, relation_id):
    mapp = {}

    for anno in data["annotations"][0]["result"]:
        if anno["type"] == "labels":
            label_id += 1
            item["entities"].append(
                {
                    "id": label_id,
                    "label": anno["value"]["labels"][0],
                    "start_offset": anno["value"]["start"],
                    "end_offset": anno["value"]["end"],
                }
            )
            mapp[anno["id"]] = label_id

    for anno in data["annotations"][0]["result"]:
        if anno["type"] == "relation":
            relation_id += 1
            item["relations"].append(
                {
                    "id": relation_id,
                    "from_id": mapp[anno["from_id"]],
                    "to_id": mapp[anno["to_id"]],
                    "type": anno["labels"][0],
                }
            )

    return item, label_id, relation_id


def convert(dataset, task_type):
    results = []
    outer_id = 0
    if task_type == "ext":
        label_id = 0
        relation_id = 0
        for data in dataset:
            if data["annotations"][0]["result"]:
                data["annotations"][0]["result"] = sorted(data["annotations"][0]["result"], key=lambda item: item["value"]["start"])
            outer_id += 1
            item = {"id": outer_id, "text": data["data"]["jfull_compress"], "entities": [], "relations": []}
            item, label_id, relation_id = append_attrs(data, item, label_id, relation_id)
            results.append(item)
    # for the classification task
    else:
        for data in dataset:
            outer_id += 1
            results.append(
                {
                    "id": outer_id,
                    "text": data["data"]["jfull_compress"],
                    "label": data["annotations"][0]["result"][0]["value"]["choices"],
                }
            )
    return results


def do_convert(args):
    if not os.path.exists(args.label_studio_file):
        raise ValueError("Please input the correct path of label studio file.")

    with open(args.label_studio_file, "r", encoding="utf-8") as fp:
        infile = fp.read()
        if len(infile) == 0:
            raise ValueError(f"Data not found in file: {args.label_studio_file}")
        dataset = json.loads(infile)
        results = convert(dataset, args.task_type)

    return results


def do_split(args, converted_results):
    set_seed(args.seed)

    tic_time = time.time()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if len(args.split_ratio) != 0 and len(args.split_ratio) != 3:
        raise ValueError("Only []/ len(split_ratio)==3 accepted for split_ratio.")

    def _check_sum(split_ratio):
        return Decimal(str(split_ratio[0])) + Decimal(str(split_ratio[1])) + Decimal(str(split_ratio[2])) == Decimal("1")

    if len(args.split_ratio) == 3 and not _check_sum(args.split_ratio):
        raise ValueError("Please set correct split_ratio, sum of elements in split_ratio should be equal to 1.")

    raw_examples = converted_results

    def _create_ext_examples(
        examples,
        negative_ratio,
        prompt_prefix="情感倾向",
        options=["正向", "负向"],
        separator="##",
        shuffle=False,
        is_train=True,
        schema_lang="ch",
    ):
        entities, relations, aspects = convert_ext_examples(
            examples, negative_ratio, prompt_prefix, options, separator, is_train, schema_lang
        )
        examples = entities + relations + aspects
        if shuffle:
            indexes = np.random.permutation(len(examples))
            examples = [examples[i] for i in indexes]
        return examples

    def _create_cls_examples(examples, prompt_prefix, options, shuffle=False):
        examples = convert_cls_examples(examples, prompt_prefix, options)
        if shuffle:
            indexes = np.random.permutation(len(examples))
            examples = [examples[i] for i in indexes]
        return examples

    def _save_examples(save_dir, file_name, examples):
        count = 0
        save_path = os.path.join(save_dir, file_name)
        with open(save_path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                count += 1
        logger.info("Save %d examples to %s." % (count, save_path))

    if len(args.split_ratio) == 0:
        if args.task_type == "ext":
            examples = _create_ext_examples(
                raw_examples,
                args.negative_ratio,
                args.prompt_prefix,
                args.options,
                args.separator,
                args.is_shuffle,
                schema_lang=args.schema_lang,
            )
        else:
            examples = _create_cls_examples(raw_examples, args.prompt_prefix, args.options, args.is_shuffle)
        _save_examples(args.save_dir, "train.txt", examples)
    else:
        if args.is_shuffle:
            indexes = np.random.permutation(len(raw_examples))
            raw_examples = [raw_examples[i] for i in indexes]

        i1, i2, _ = args.split_ratio

        p1 = int(len(raw_examples) * i1)
        p2 = int(len(raw_examples) * (i1 + i2))

        if args.task_type == "ext":
            train_examples = _create_ext_examples(
                raw_examples[:p1],
                args.negative_ratio,
                args.prompt_prefix,
                args.options,
                args.separator,
                args.is_shuffle,
                schema_lang=args.schema_lang,
            )
            dev_examples = _create_ext_examples(
                raw_examples[p1:p2],
                -1,
                args.prompt_prefix,
                args.options,
                args.separator,
                is_train=False,
                schema_lang=args.schema_lang,
            )
            test_examples = _create_ext_examples(
                raw_examples[p2:],
                -1,
                args.prompt_prefix,
                args.options,
                args.separator,
                is_train=False,
                schema_lang=args.schema_lang,
            )
        else:
            train_examples = _create_cls_examples(raw_examples[:p1], args.prompt_prefix, args.options)
            dev_examples = _create_cls_examples(raw_examples[p1:p2], args.prompt_prefix, args.options)
            test_examples = _create_cls_examples(raw_examples[p2:], args.prompt_prefix, args.options)

        if not train_examples:
            train_examples = dev_examples if dev_examples else test_examples

        if not dev_examples:
            dev_examples = test_examples

        _save_examples(args.save_dir, "train.txt", train_examples)
        _save_examples(args.save_dir, "dev.txt", dev_examples)
        _save_examples(args.save_dir, "test.txt", test_examples)

    logger.info("Finished! It takes %.2f seconds" % (time.time() - tic_time))


def main(args: dict):
    # Label studio data preparation
    label_studio_args = LabelStudioArgs(**args)
    logger.info("Preparing dataset from label studio data...")

    # Convert label studio data to the format of model inputs
    converted_results = do_convert(label_studio_args)

    # Check spaces
    logger.info("Checking if invalid spaces exist in texts...")
    for data in tqdm(converted_results):
        check_spaces(data['text'])

    # Split data as train, dev, test
    do_split(label_studio_args, converted_results)
