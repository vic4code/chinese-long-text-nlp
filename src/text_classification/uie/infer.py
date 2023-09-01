import json
import os
from dataclasses import dataclass, field
from typing import List

from paddlenlp import Taskflow
from paddlenlp.transformers import AutoTokenizer
from paddlenlp.utils.log import logger
from tqdm import tqdm

from ..utc.utils import get_template_tokens_len
from .utils import RuleBasedProcessor, filter_text


@dataclass
class InferArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `PdArgumentParser` we can turn this class into argparse arguments to be able to
    specify them on the command line.
    """

    # Data args
    label_file: str = field(
        default="./data/modeling/text_classification/label_studio/labels.txt",
        metadata={"help": "The path of the labels defined in the classification task."},
    )
    data_file_or_path_to_inference: str = field(
        default="./data/inference/infer_example.json",
        metadata={"help": "The inference json file or the path of utc dataset to inference."},
    )
    max_seq_length: int = field(
        default=768, metadata={'description': 'Max sequence length of a text sequence input to the pretrained model.'}
    )

    # Model args
    schema: List[str] = field(default_factory=list, metadata={'description': 'UIE schema.'})
    utc_model_name_or_path: str = field(
        default="utc-base", metadata={"help": "The model name or the checkpoint path of the utc pretrained model from PaddleNLP."}
    )
    uie_model_name_or_path: str = field(
        default="./models/text_classification/uie/checkpoints/model_best",
        metadata={"help": "The model name or the checkpoint path of the uie pretrained model from PaddleNLP."},
    )
    threshold: float = field(default=0.4, metadata={'description': 'Threshold to decide whether to output the results or not.'})
    dynamic_adjust_length: bool = field(default=True, metadata={'description': 'If adjust the sequence length dynamically or not.'})


def uie_process(args, model, processer, text, max_content_len):
    """
    Process text with uie model.
    """
    uie_output = model(text)
    uie_output = processer.postprocessing(raw_text=text, uie_output=uie_output, schema=args.schema)
    new_text = filter_text(
        raw_text=text,
        uie_output=uie_output,
        max_len_of_new_text=max_content_len,
        threshold=args.threshold,
        dynamic_adjust_length=args.dynamic_adjust_length,
    )

    if len(new_text) == 0:
        new_text = text[: args.max_content_len]

    return new_text


def inference(args):
    """
    Shorten long texts .json file or output train.txt, dev.txt, test.txt with the finetuned uie model for utc model.
    """
    # setting
    uie = Taskflow("information_extraction", task_path=args.uie_model_name_or_path, schema=args.schema, precision="fp16")
    tokenizer = AutoTokenizer.from_pretrained(args.utc_model_name_or_path)
    special_word_len = get_template_tokens_len(tokenizer, args.label_file)
    max_content_len = args.max_seq_length - special_word_len

    # Check if the path and files valid
    is_valid_json = os.path.isfile(args.data_file_or_path_to_inference) and args.data_file_or_path_to_inference.endswith('.json')
    has_required_files = all(
        [os.path.isfile(os.path.join(args.data_file_or_path_to_inference, data_name)) for data_name in ['train.txt', 'dev.txt', 'test.txt']]
    )

    if not (is_valid_json or has_required_files):
        logger.error(f"{args.data_file_or_path_to_inference} is not a valid file or path to inference.")
        raise ValueError(
            "Invalid file or path. The input json file must be exported from CA verdict database and with 'jfull_compress' included."
            "If a path is given, it must include train.txt, dev.txt, test.txt files which are generated from utc_dataprep."
        )

    # Process utc inference data
    if args.data_file_or_path_to_inference.endswith('.json'):
        # Read
        with open(args.data_file_or_path_to_inference, "r", encoding="utf8") as fp:
            data = json.load(fp)
            number_of_examples = len(data)

        logger.info(f"Start preprocessing {args.data_file_or_path_to_inference}...")
        for example in tqdm(data, total=number_of_examples):
            new_text = uie_process(
                args=args, model=uie, processer=RuleBasedProcessor(), text=example["jfull_compress"], max_content_len=max_content_len
            )
            example["jfull_compress"] = new_text

        # Write
        filename = args.data_file_or_path_to_inference.split("/")[-1]
        with open(
            os.path.join(args.data_file_or_path_to_inference.replace(filename, "uie_processed_data.json")), "w", encoding="utf-8"
        ) as fp:
            json.dump(data, fp, ensure_ascii=False, indent=4)

        logger.info(f"Finish inference_data preprocessing. Total samples: {len(data)}.")

    # Process utc training data
    else:
        total = 0
        for data_name in ['train.txt', 'dev.txt', 'test.txt']:
            out_text = []
            logger.info(f"Start preprocessing {data_name}...")
            number_of_examples = 0

            with open(os.path.join(args.data_file_or_path_to_inference, data_name), "r", encoding="utf8") as fp:
                number_of_examples = len(fp.readlines())
            total += number_of_examples

            # Read
            with open(os.path.join(args.data_file_or_path_to_inference, data_name), "r", encoding="utf8") as fp:
                for example in tqdm(fp, total=number_of_examples):
                    example = json.loads(example.strip())
                    new_text = uie_process(
                        args=args,
                        model=uie,
                        processer=RuleBasedProcessor(),
                        text=example["text_a"],
                        max_content_len=max_content_len,
                        # labels=example["labels"],
                    )
                    example["text_a"] = new_text
                    out_text.append(example)

            # Write
            the_last_path = args.data_file_or_path_to_inference.split("/")[-1]
            output_path = args.data_file_or_path_to_inference.replace(the_last_path, "uie_processed_dataset")

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            with open(os.path.join(output_path, data_name), "w", encoding="utf-8") as outfile:
                for text in out_text:
                    jsonString = json.dumps(text, ensure_ascii=False)
                    outfile.write(jsonString)
                    outfile.write("\n")

            logger.info(f"Finish {data_name} processing. Total samples: {len(out_text)}.")

    logger.info("Finish all preprocessing.")


def main(args: dict):
    infer_args = InferArgs(**args)
    inference(infer_args)
