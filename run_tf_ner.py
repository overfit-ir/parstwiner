#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
""" Fine-tuning the library models for named entity recognition."""


import logging
import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple

import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

from transformers import (
    AutoConfig,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TFAutoModelForTokenClassification,
    TFTrainer,
    TFTrainingArguments,
)
from transformers.utils import logging as hf_logging
from utils_ner import Split, TFTokenClassificationDataset, TokenClassificationTask, MultitaskModel, MultitaskTrainer


hf_logging.set_verbosity_info()
hf_logging.enable_default_handler()
hf_logging.enable_explicit_format()


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={
                           "help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArgumentsTwitter:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir_twitter: str = field(
        metadata={
            "help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels_twitter: Optional[str] = field(
        metadata={
            "help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    max_seq_length_twitter: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache_twitter: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class DataTrainingArgumentsPeyma:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir_peyma: str = field(
        metadata={
            "help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels_peyma: Optional[str] = field(
        metadata={
            "help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."}
    )
    max_seq_length_peyma: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache_peyma: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(
        (ModelArguments,
         DataTrainingArgumentsTwitter,
         DataTrainingArgumentsPeyma,
         TFTrainingArguments))
    model_args, data_args_twitter, data_args_peyma, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    module = import_module("tasks")

    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task_twitter: TokenClassificationTask = token_classification_task_clazz()
        token_classification_task_peyma: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(
        "n_replicas: %s, distributed training: %s, 16-bits training: %s",
        training_args.n_replicas,
        bool(training_args.n_replicas > 1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Prepare Token Classification task
    labels_twitter = token_classification_task_twitter.get_labels(
        data_args_twitter.labels_twitter)
    label_map_twitter: Dict[int, str] = {
        i: label for i, label in enumerate(labels_twitter)}
    num_labels_twitter = len(labels_twitter)

    labels_peyma = token_classification_task_peyma.get_labels(
        data_args_peyma.labels_peyma)
    label_map_peyma: Dict[int, str] = {
        i: label for i, label in enumerate(labels_peyma)}
    num_labels_peyma = len(labels_peyma)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     id2label=label_map,
    #     label2id={label: i for i, label in enumerate(labels)},
    #     cache_dir=model_args.cache_dir,
    # )
    config_dict = {
        "twitter": AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels_twitter,
            id2label=label_map_twitter,
            label2id={label: i for i, label in enumerate(labels_twitter)},
            cache_dir=model_args.cache_dir,
        ),
        "peyma": AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels_peyma,
            id2label=label_map_peyma,
            label2id={label: i for i, label in enumerate(labels_peyma)},
            cache_dir=model_args.cache_dir,
        ),
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )

    with training_args.strategy.scope():
        # model = TFAutoModelForTokenClassification.from_pretrained(
        #     model_args.model_name_or_path,
        #     from_pt=bool(".bin" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        # )

        multitask_model = MultitaskModel.create(
            model_name=model_args.model_name_or_path,
            model_type_dict={
                "twitter": TFAutoModelForTokenClassification,
                "peyma": TFAutoModelForTokenClassification,
            },
            model_config_dict=config_dict,
        )

    # Get datasets
    train_dataset_twitter = (
        TFTokenClassificationDataset(
            token_classification_task=token_classification_task_twitter,
            data_dir=data_args_twitter.data_dir_twitter,
            tokenizer=tokenizer,
            labels=labels_twitter,
            model_type=config_dict['twitter'].model_type,
            max_seq_length=data_args_twitter.max_seq_length_twitter,
            overwrite_cache=data_args_twitter.overwrite_cache_twitter,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    train_dataset_peyma = (
        TFTokenClassificationDataset(
            token_classification_task=token_classification_task_peyma,
            data_dir=data_args_peyma.data_dir_peyma,
            tokenizer=tokenizer,
            labels=labels_peyma,
            model_type=config_dict['peyma'].model_type,
            max_seq_length=data_args_peyma.max_seq_length_peyma,
            overwrite_cache=data_args_peyma.overwrite_cache_peyma,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset_twitter = (
        TFTokenClassificationDataset(
            token_classification_task=token_classification_task_twitter,
            data_dir=data_args_twitter.data_dir_twitter,
            tokenizer=tokenizer,
            labels=labels_twitter,
            model_type=config_dict['twitter'].model_type,
            max_seq_length=data_args_twitter.max_seq_length_twitter,
            overwrite_cache=data_args_twitter.overwrite_cache_twitter,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    eval_dataset_peyma = (
        TFTokenClassificationDataset(
            token_classification_task=token_classification_task_peyma,
            data_dir=data_args_peyma.data_dir_peyma,
            tokenizer=tokenizer,
            labels=labels_peyma,
            model_type=config_dict['peyma'].model_type,
            max_seq_length=data_args_peyma.max_seq_length_peyma,
            overwrite_cache=data_args_peyma.overwrite_cache_peyma,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != -100:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        return preds_list, out_label_list

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list = align_predictions(
            p.predictions, p.label_ids)

        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }

    train_dataset_dict = {
        'twitter': train_dataset_twitter,
        'peyma': train_dataset_peyma,
    }

    eval_dataset_dict = {
        'twitter': eval_dataset_twitter,
        'peyma': eval_dataset_peyma,
    }
    # Initialize our Trainer
    trainer = MultitaskTrainer(
        model=multitask_model,
        args=training_args,
        train_dataset=train_dataset_dict,
        eval_dataset=eval_dataset_twitter,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()
        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")

            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        test_dataset = TFTokenClassificationDataset(
            token_classification_task=token_classification_task_twitter,
            data_dir=data_args_twitter.data_dir_twitter,
            tokenizer=tokenizer,
            labels=labels_twitter,
            model_type=config_dict['twitter'].model_type,
            max_seq_length=data_args_twitter.max_seq_length_twitter,
            overwrite_cache=data_args_twitter.overwrite_cache_twitter,
            mode=Split.test,
        )

        predictions, label_ids, metrics = trainer.predict(
            test_dataset.get_dataset())
        preds_list, labels_list = align_predictions(predictions, label_ids)
        report = classification_report(labels_list, preds_list)

        logger.info("\n%s", report)

        output_test_results_file = os.path.join(
            training_args.output_dir, "test_results.txt")

        with open(output_test_results_file, "w") as writer:
            writer.write("%s\n" % report)

        # Save predictions
        output_test_predictions_file = os.path.join(
            training_args.output_dir, "test_predictions.txt")

        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
                example_id = 0

                for line in f:
                    if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                        writer.write(line)

                        if not preds_list[example_id]:
                            example_id += 1
                    elif preds_list[example_id]:
                        output_line = line.split(
                        )[0] + " " + preds_list[example_id].pop(0) + "\n"

                        writer.write(output_line)
                    else:
                        logger.warning(
                            "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])

    return results


if __name__ == "__main__":
    main()
