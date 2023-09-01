import os
from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple, Union

from paddle import optimizer, set_device
from paddle.static import InputSpec
from paddlenlp.datasets import load_dataset
from paddlenlp.trainer import Trainer, TrainingArguments, get_last_checkpoint
from paddlenlp.transformers import UIE, AutoTokenizer, export_model

from . import BASE_CONFIG_PATH, load_config, logger
from .utils.data_utils import convert_to_uie_format, read_data_by_chunk
from .utils.model_utils import compute_metrics, uie_loss_func


def finetune(
    dataset_path: str,
    train_file: str,
    dev_file: str = None,
    test_file: str = None,
    max_seq_len: int = 512,
    model_name_or_path: str = "uie-base",
    export_model_dir: Optional[str] = None,
    convert_and_tokenize_function: Optional[Callable[[Dict[str, str], Any, int], Dict[str, Union[str, float]]]] = convert_to_uie_format,
    criterion=uie_loss_func,
    compute_metrics=compute_metrics,
    optimizers: Optional[Tuple[optimizer.Optimizer, optimizer.lr.LRScheduler]] = (None, None),
    training_args: Optional[TrainingArguments] = None,
) -> None:
    train_path, dev_path, test_path = (os.path.join(dataset_path, file) for file in (train_file, dev_file, test_file))
    working_data = {"train": None}
    if not os.path.exists(dev_path):
        if training_args.do_eval:
            logger.warning(
                f"Evaluation data not found in {dev_path}. \
                Please input the correct path of evaluation data.\
                    Auto-training without evaluation data..."
            )
        training_args.do_eval = False
    else:
        working_data["dev"] = None
    if not os.path.exists(test_path):
        if training_args.do_predict:
            logger.warning(
                f"Testing data not found in {test_path}. \
                Please input the correct path of testing data.\
                    Auto-training without testing data..."
            )
        training_args.do_predict = False
    else:
        working_data["test"] = None

    if training_args.load_best_model_at_end and not training_args.do_eval:
        raise ValueError(
            "Cannot load best model at end when do_eval is False. Auto-adjust. " + "Please adjust load_best_model_at_end or do_eval."
        )

    # Model & Data Setup
    set_device(training_args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = UIE.from_pretrained(model_name_or_path)
    convert_function = partial(
        convert_and_tokenize_function,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )
    for data in working_data:
        working_data[data] = load_dataset(
            read_data_by_chunk,
            data_path=eval(f"{data}_path"),
            max_seq_len=max_seq_len,
            lazy=False,
        )
        working_data[data] = working_data[data].map(convert_function)

    # Trainer Setup
    trainer = Trainer(
        model=model,
        criterion=criterion,
        args=training_args,
        train_dataset=working_data["train"] if training_args.do_train else None,
        eval_dataset=working_data["dev"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        optimizers=optimizers,
    )
    trainer.optimizers = (
        optimizer.AdamW(learning_rate=training_args.learning_rate, parameters=model.parameters())
        if optimizers[0] is None
        else optimizers[0]
    )

    # Checkpoint Setup
    checkpoint, last_checkpoint = None, None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. " "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # Start Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Start Evaluate and tests model
    if training_args.do_eval:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)

    # Start Testing
    if training_args.do_predict:
        predict_output = trainer.predict(test_dataset=working_data["test"])
        trainer.log_metrics("test", predict_output.metrics)

    # export inference model
    if training_args.do_export:
        export_model_dir = export_model_dir if export_model_dir else training_args.output_dir
        export_model(
            model=trainer.model,
            input_spec=[
                InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
                InputSpec(shape=[None, None], dtype="int64", name="token_type_ids"),
                InputSpec(shape=[None, None], dtype="int64", name="position_ids"),
                InputSpec(shape=[None, None], dtype="int64", name="attention_mask"),
            ],
            path=export_model_dir,
        )
        trainer.tokenizer.save_pretrained(export_model_dir)
    logger.info("Finish training.")


def main(args: dict):
    training_args = TrainingArguments(**args["training_args"])
    model_args, data_args = args["model_args"], args["data_args"]
    train_path = os.path.join(data_args["dataset_path"], data_args["train_file"])
    if not os.path.exists(train_path):
        logger.warning(f"Training data not found in {train_path}. Automatically Converting...")
        convert_args = load_config(yaml_file=os.path.join(BASE_CONFIG_PATH, "dataprep.yaml"))
        from .dataprep import main as main_convert

        main_convert(args=convert_args)

    finetune(
        dataset_path=data_args["dataset_path"],
        train_file=data_args["train_file"],
        dev_file=data_args["dev_file"],
        test_file=data_args["test_file"],
        max_seq_len=model_args["max_seq_len"],
        model_name_or_path=model_args["model_name_or_path"],
        export_model_dir=data_args["export_model_dir"],
        training_args=training_args,
    )
