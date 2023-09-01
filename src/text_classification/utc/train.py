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

from dataclasses import dataclass, field

import paddle
from paddle.static import InputSpec
from paddlenlp.datasets import load_dataset
from paddlenlp.prompt import PromptModelForSequenceClassification, PromptTrainer, PromptTuningArguments, UTCTemplate
from paddlenlp.transformers import UTC, AutoTokenizer, export_model

from .metric import MetricReport
from .utils import UTCLoss, read_local_dataset


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="./data/modeling/text_classification/model_train/utc/dataset",
        metadata={"help": "Local dataset directory including train.txt, dev.txt."},
    )
    train_file: str = field(default="train.txt", metadata={"help": "Train dataset file name."})
    dev_file: str = field(default="dev.txt", metadata={"help": "Dev dataset file name."})


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="utc-base",
        metadata={
            "help": "The build-in pretrained UTC model name or path to its checkpoints, such as "
            "`utc-xbase`, `utc-base`, `utc-medium`, `utc-mini`, `utc-micro`, `utc-nano` and `utc-pico`."
        },
    )
    export_type: str = field(default="paddle", metadata={"help": "The type to export. Support `paddle` and `onnx`."})


def finetune(data_args, training_args, model_args) -> None:
    paddle.set_device(training_args.device)

    # Load the pretrained language model.
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = UTC.from_pretrained(model_args.model_name_or_path)

    # Define template for preprocess and verbalizer for postprocess.
    template = UTCTemplate(tokenizer, training_args.max_seq_length)

    # Load and preprocess dataset.
    train_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_path,
        data_file=data_args.train_file,
        max_seq_len=training_args.max_seq_length,
        lazy=False,
    )
    dev_ds = load_dataset(
        read_local_dataset,
        data_path=data_args.dataset_path,
        data_file=data_args.dev_file,
        lazy=False,
    )

    # Define the criterion.
    criterion = UTCLoss()

    # Initialize the prompt model.
    prompt_model = PromptModelForSequenceClassification(
        model, template, None, freeze_plm=training_args.freeze_plm, freeze_dropout=training_args.freeze_dropout
    )

    # Define the metric function.
    def compute_metrics(eval_preds):
        metric = MetricReport()
        preds = paddle.to_tensor(eval_preds.predictions)
        metric.reset()
        metric.update(preds, paddle.to_tensor(eval_preds.label_ids))
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
        args=training_args,
        criterion=criterion,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        callbacks=None,
        compute_metrics=compute_metrics,
    )

    # Training.
    if training_args.do_train:
        train_results = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        metrics = train_results.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)

    # Export.
    if training_args.do_export:
        input_spec = [
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
            InputSpec(shape=[None, None, None, None], dtype="float32", name="attention_mask"),
            InputSpec(shape=[None, None], dtype="int64", name="omask_positions"),
            InputSpec(shape=[None], dtype="int64", name="cls_positions"),
        ]
        export_model(trainer.pretrained_model, input_spec, training_args.output_dir, model_args.export_type)


def main(args: dict):
    # Parse the arguments.
    model_args, data_args, training_args = (
        ModelArguments(**args["model_args"]),
        DataArguments(**args["data_args"]),
        PromptTuningArguments(**args["training_args"]),
    )

    training_args.learning_rate = float(training_args.learning_rate)
    training_args.adam_epsilon = float(training_args.adam_epsilon)

    if model_args.model_name_or_path in ["utc-base", "utc-large"]:
        model_args.multilingual = True

    # Log model and data config
    training_args.print_config(model_args, "Model")
    training_args.print_config(data_args, "Data")

    # Train and validate
    finetune(model_args=model_args, data_args=data_args, training_args=training_args)
