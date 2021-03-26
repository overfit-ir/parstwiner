# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock
from transformers import (
    PreTrainedTokenizer,
    is_tf_available,
    is_torch_available,
    TFPreTrainedModel,
    TFTrainer,
    PretrainedConfig,
)


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class TokenClassificationTask:
    @staticmethod
    def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
        raise NotImplementedError

    @staticmethod
    def get_labels(path: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ) -> List[InputFeatures]:
        """Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        # TODO clean up all this to leverage built-in features of tokenizers

        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10_000 == 0:
                logger.info("Writing example %d of %d",
                            ex_index, len(examples))

            tokens = []
            label_ids = []
            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)

                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend(
                        [label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (
                    max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1]
                              * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] *
                               padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * \
                    padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join(
                    [str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join(
                    [str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join(
                    [str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join(
                    [str(x) for x in label_ids]))

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
                )
            )
        return features


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class TokenClassificationDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}".format(
                    mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(
                        f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(
                        f"Creating features from dataset file at {data_dir}")
                    examples = token_classification_task.read_examples_from_file(
                        data_dir, mode)
                    # TODO clean up all this to leverage built-in features of tokenizers
                    self.features = token_classification_task.convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in [
                            "xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                    logger.info(
                        f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf
    from tensorflow.data import Dataset
    import numpy as np
    import math
    from transformers.trainer_utils import PredictionOutput, EvalPrediction
    from typing import Callable, Dict, Optional, Tuple

    class MultitaskModel(TFPreTrainedModel):
        def __init__(self, encoder, taskmodels_dict):
            """
            Setting MultitaskModel up as a PretrainedModel allows us
            to take better advantage of Trainer features
            """
            super().__init__(PretrainedConfig())

            self.encoder = encoder
            self.taskmodels_dict = taskmodels_dict

        @classmethod
        def create(cls, model_name, model_type_dict, model_config_dict):
            """
            This creates a MultitaskModel using the model class and config objects
            from single-task models.

            We do this by creating each single-task model, and having them share
            the same encoder transformer.
            """
            shared_encoder = None
            taskmodels_dict = {}
            for task_name, model_type in model_type_dict.items():
                model = model_type.from_pretrained(
                    model_name,
                    config=model_config_dict[task_name],
                )
                if shared_encoder is None:
                    shared_encoder = getattr(
                        model, cls.get_encoder_attr_name(model))
                else:
                    setattr(model, cls.get_encoder_attr_name(
                        model), shared_encoder)
                taskmodels_dict[task_name] = model
            return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

        @classmethod
        def get_encoder_attr_name(cls, model):
            """
            The encoder transformer is named differently in each model "architecture".
            This method lets us get the name of the encoder attribute
            """
            model_class_name = model.__class__.__name__
            if model_class_name.startswith("TFBert"):
                return "bert"
            elif model_class_name.startswith("TFRoberta"):
                return "roberta"
            elif model_class_name.startswith("TFAlbert"):
                return "albert"
            else:
                raise KeyError(f"Add support for new model {model_class_name}")

        def __call__(self, task_name, *args, **kwargs):
            return self.taskmodels_dict[task_name](*args, **kwargs)

    class StrIgnoreDevice(str):
        """
        This is a hack. The Trainer is going call .to(device) on every input
        value, but we need to pass in an additional `task_name` string.
        This prevents it from throwing an error
        """

        def to(self, device):
            return self

    class TFDatasetWithTaskname:
        """
        Wrapper around a DataLoader to also yield a task name
        """

        def __init__(self, task_name, tf_dataset, length):
            self.task_name = task_name
            self.tf_dataset = tf_dataset
            self.length = length
            # self.batch_size = tf_dataset.batch_size
            # self.dataset = tf_dataset.dataset

        def __len__(self):
            return self.length

        def __iter__(self):
            for batch in self.tf_dataset:
                # batch["task_name"] = self.task_name
                batch_dict = {}
                batch_dict['batch'] = batch
                batch_dict['task_name'] = self.task_name
                yield batch_dict

    class MultitaskTFDataset:
        """
        Data loader that combines and samples from multiple single-task
        data loaders.
        """

        def __init__(self, dataloader_dict, batch_size, approx):
            self.dataloader_dict = dataloader_dict

            self.num_batches_dict = {
                task_name: approx(len(dataloader) / batch_size)
                for task_name, dataloader in self.dataloader_dict.items()
            }
            self.task_name_list = list(self.dataloader_dict)
            # self.dataset = [None] * sum(
            #     len(dataloader.dataset)
            #     for dataloader in self.dataloader_dict.values()
            # )

        def __len__(self):
            return sum(self.num_batches_dict.values())

        def __iter__(self):
            """
            For each batch, sample a task, and yield a batch from the respective
            task Dataloader.

            We use size-proportional sampling, but you could easily modify this
            to sample from some-other distribution.
            """
            task_choice_list = []
            for i, task_name in enumerate(self.task_name_list):
                task_choice_list += [i] * self.num_batches_dict[task_name]
            task_choice_list = np.array(task_choice_list)
            np.random.shuffle(task_choice_list)
            dataloader_iter_dict = {
                task_name: iter(dataloader)
                for task_name, dataloader in self.dataloader_dict.items()
            }
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            print(len(task_choice_list), sum(task_choice_list))
            print(self.task_name_list)
            print(self.dataloader_dict)
            for task_choice in task_choice_list:
                task_name = self.task_name_list[task_choice]
                yield next(dataloader_iter_dict[task_name])

    class MultitaskTrainer(TFTrainer):

        def run_model(self, features, labels, task_name, training):
            """
            Computes the loss of the given features and labels pair.
            Subclass and override this method if you want to inject some custom behavior.
            Args:
                features (:obj:`tf.Tensor`): A batch of input features.
                labels (:obj:`tf.Tensor`): A batch of labels.
                training (:obj:`bool`): Whether or not to run the model in training mode.
            Returns:
                A tuple of two :obj:`tf.Tensor`: The loss and logits.
            """

            if self.args.past_index >= 0 and getattr(self, "_past", None) is not None:
                features["mems"] = self._past

            if isinstance(labels, (dict)):
                outputs = self.model(task_name, features,
                                     training=training, **labels)[:2]
            else:
                outputs = self.model(task_name, features,
                                     labels=labels, training=training)[:2]

            loss, logits = outputs[:2]

            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            return loss, logits

        def training_step(self, features, labels, nb_instances_in_global_batch, task_name):
            """
            Perform a training step on features and labels.
            Subclass and override to inject some custom behavior.
            """
            per_example_loss, _ = self.run_model(
                features, labels, task_name, True)
            scaled_loss = per_example_loss / \
                tf.cast(nb_instances_in_global_batch,
                        dtype=per_example_loss.dtype)
            gradients = tf.gradients(
                scaled_loss, self.model.trainable_variables)
            gradients = [
                g if g is not None else tf.zeros_like(v) for g, v in zip(gradients, self.model.trainable_variables)
            ]

            if self.args.gradient_accumulation_steps > 1:
                self.gradient_accumulator(gradients)

            self.train_loss.update_state(scaled_loss)

            if self.args.gradient_accumulation_steps == 1:
                return gradients

        def apply_gradients(self, features, labels, nb_instances_in_global_batch, task_name):
            if self.args.gradient_accumulation_steps == 1:
                gradients = self.training_step(
                    features, labels, nb_instances_in_global_batch, task_name)

                self.optimizer.apply_gradients(
                    list(zip(gradients, self.model.trainable_variables)))
            else:
                raise NotImplementedError(
                    'gradient_accumulation_steps must be equal to 1')

        @tf.function
        def distributed_training_steps(self, batch):
            with self.args.strategy.scope():

                nb_instances_in_batch = self._compute_nb_instances(
                    batch['batch'])
                batch_inputs = self._get_step_inputs(
                    batch['batch'], nb_instances_in_batch)
                task_name = batch['task_name']
                features, labels, nb_instances = batch_inputs
                inputs = features, labels, nb_instances, task_name

                self.args.strategy.run(self.apply_gradients, inputs)

        @tf.function
        def distributed_prediction_steps(self, batch):

            nb_instances_in_batch = self._compute_nb_instances(batch['batch'])
            batch_inputs = self._get_step_inputs(
                batch['batch'], nb_instances_in_batch)
            features, labels, nb_instances = batch_inputs
            task_name = batch['task_name']
            inputs = features, labels, nb_instances, task_name

            logits = self.args.strategy.run(self.prediction_step, inputs)

            return logits

        def prediction_step(
            self, features: tf.Tensor, labels: tf.Tensor, nb_instances_in_global_batch: tf.Tensor, task_name
        ) -> tf.Tensor:
            """
            Compute the prediction on features and update the loss with labels.
            Subclass and override to inject some custom behavior.
            """
            per_example_loss, logits = self.run_model(
                features, labels, task_name, False)
            scaled_loss = per_example_loss / \
                tf.cast(nb_instances_in_global_batch,
                        dtype=per_example_loss.dtype)

            self.eval_loss.update_state(scaled_loss)

            return logits

        def prediction_loop(
            self,
            dataset: tf.data.Dataset,
            steps: int,
            num_examples: int,
            description: str,
            prediction_loss_only: Optional[bool] = None,
        ) -> PredictionOutput:
            """
            Prediction/evaluation loop, shared by :func:`~transformers.TFTrainer.evaluate` and
            :func:`~transformers.TFTrainer.predict`.
            Works both with or without labels.
            """

            prediction_loss_only = (
                prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
            )

            logger.info("***** Running %s *****", description)
            logger.info("  Num examples in dataset = %d", num_examples)
            if description == "Evaluation":
                logger.info("  Num examples in used in evaluation = %d",
                            self.args.eval_batch_size * steps)
            logger.info("  Batch size = %d", self.args.eval_batch_size)

            label_ids: np.ndarray = None
            preds: np.ndarray = None
            self.eval_loss.reset_states()

            # Reset the past mems state at the beginning of the evaluation if necessary.
            if self.args.past_index >= 0:
                self._past = None

            for step, batch in enumerate(dataset):
                logits = self.distributed_prediction_steps(batch)
                _, labels = batch['batch']

                if not prediction_loss_only:
                    if isinstance(logits, tuple):
                        logits = logits[0]

                    if isinstance(labels, tuple):
                        labels = labels[0]

                    if self.args.n_replicas > 1:
                        for val in logits.values:
                            if preds is None:
                                preds = val.numpy()
                            else:
                                preds = np.append(preds, val.numpy(), axis=0)

                        for val in labels.values:
                            if label_ids is None:
                                label_ids = val.numpy()
                            else:
                                label_ids = np.append(
                                    label_ids, val.numpy(), axis=0)
                    else:
                        if preds is None:
                            preds = logits.numpy()
                        else:
                            preds = np.append(preds, logits.numpy(), axis=0)

                        if label_ids is None:
                            label_ids = labels.numpy()
                        else:
                            label_ids = np.append(
                                label_ids, labels.numpy(), axis=0)

                    if step == steps - 1:
                        break

            if self.compute_metrics is not None and preds is not None and label_ids is not None:
                metrics = self.compute_metrics(EvalPrediction(
                    predictions=preds, label_ids=label_ids))
            else:
                metrics = {}

            metrics["eval_loss"] = self.eval_loss.result().numpy() / steps

            for key in list(metrics.keys()):
                if not key.startswith("eval_"):
                    metrics[f"eval_{key}"] = metrics.pop(key)

            if self.args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

        def get_single_train_tfdataset(self, task_name, train_dataset):
            """
            Create a single-task data loader that also yields task names
                """
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            # self.total_train_batch_size = self.args.train_batch_size * \
            #     self.args.gradient_accumulation_steps
            num_train_examples = train_dataset.cardinality().numpy()

            if self.num_train_examples < 0:
                raise ValueError(
                    "The training dataset must have an asserted cardinality")

            ds = (
                train_dataset.repeat()
                .shuffle(num_train_examples, seed=self.args.seed)
                .batch(self.total_train_batch_size, drop_remainder=self.args.dataloader_drop_last)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

            dataset = TFDatasetWithTaskname(
                task_name=task_name,
                tf_dataset=self.args.strategy.experimental_distribute_dataset(
                    ds),
                length=num_train_examples,
            )

            return dataset

        def get_train_tfdataset(self):
            """
            Returns a MultitaskDataloader, which is not actually a Dataloader
            but an iterable that returns a generator that samples from each 
            task Dataloader
            """
            self.total_train_batch_size = self.args.train_batch_size * \
                self.args.gradient_accumulation_steps

            num_train_examples = 0
            for _, task_dataset in self.train_dataset.items():
                num_train_examples += task_dataset.get_dataset().cardinality().numpy()

            self.num_train_examples = num_train_examples
            approx = math.floor if self.args.dataloader_drop_last else math.ceil

            return MultitaskTFDataset({
                task_name: self.get_single_train_tfdataset(
                    task_name, task_dataset.get_dataset())
                for task_name, task_dataset in self.train_dataset.items()
            }, self.args.train_batch_size, approx)

        def get_eval_tfdataset(self, eval_dataset):
            """
            Returns a MultitaskDataloader, which is not actually a Dataloader
            but an iterable that returns a generator that samples from each
            task Dataloader
            """
            if eval_dataset is None and self.eval_dataset is None:
                raise ValueError(
                    "Trainer: evaluation requires an eval_dataset.")

            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            num_examples = eval_dataset.cardinality().numpy()

            if num_examples < 0:
                raise ValueError(
                    "The training dataset must have an asserted cardinality")

            approx = math.floor if self.args.dataloader_drop_last else math.ceil
            steps = approx(num_examples / self.args.eval_batch_size)

            ds = (
                eval_dataset.repeat()
                .batch(self.args.eval_batch_size, drop_remainder=self.args.dataloader_drop_last)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

            dataset = TFDatasetWithTaskname(
                task_name='twitter',
                tf_dataset=self.args.strategy.experimental_distribute_dataset(
                    ds),
                length=num_examples,
            )
            return dataset, steps, num_examples

        def get_test_tfdataset(self, test_dataset):
            """
            Returns a test :class:`~tf.data.Dataset`.
            Args:
                test_dataset (:class:`~tf.data.Dataset`):
                    The dataset to use. The dataset should yield tuples of ``(features, labels)`` where ``features`` is a
                    dict of input features and ``labels`` is the labels. If ``labels`` is a tensor, the loss is calculated
                    by the model by calling ``model(features, labels=labels)``. If ``labels`` is a dict, such as when using
                    a QuestionAnswering head model with multiple targets, the loss is instead calculated by calling
                    ``model(features, **labels)``.
            Subclass and override this method if you want to inject some custom behavior.
            """

            num_examples = test_dataset.cardinality().numpy()

            if num_examples < 0:
                raise ValueError(
                    "The training dataset must have an asserted cardinality")

            steps = math.ceil(num_examples / self.args.eval_batch_size)
            ds = test_dataset.batch(self.args.eval_batch_size).prefetch(
                tf.data.experimental.AUTOTUNE)

            dataset = TFDatasetWithTaskname(
                task_name='twitter',
                tf_dataset=self.args.strategy.experimental_distribute_dataset(
                    ds),
                length=num_examples,
            )

            return dataset, steps, num_examples

    class TFTokenClassificationDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = -100
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            examples = token_classification_task.read_examples_from_file(
                data_dir, mode)
            # TODO clean up all this to leverage built-in features of tokenizers
            self.features = token_classification_task.convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token_at_end=bool(model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(tokenizer.padding_side == "left"),
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
            )

            def gen():
                for ex in self.features:
                    if ex.token_type_ids is None:
                        yield (
                            {"input_ids": ex.input_ids,
                                "attention_mask": ex.attention_mask},
                            ex.label_ids,
                        )
                    else:
                        yield (
                            {
                                "input_ids": ex.input_ids,
                                "attention_mask": ex.attention_mask,
                                "token_type_ids": ex.token_type_ids,
                            },
                            ex.label_ids,
                        )

            if "token_type_ids" not in tokenizer.model_input_names:
                self.dataset = tf.data.Dataset.from_generator(
                    gen,
                    ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                    (
                        {"input_ids": tf.TensorShape(
                            [None]), "attention_mask": tf.TensorShape([None])},
                        tf.TensorShape([None]),
                    ),
                )
            else:
                self.dataset = tf.data.Dataset.from_generator(
                    gen,
                    ({"input_ids": tf.int32, "attention_mask": tf.int32,
                      "token_type_ids": tf.int32}, tf.int64),
                    (
                        {
                            "input_ids": tf.TensorShape([None]),
                            "attention_mask": tf.TensorShape([None]),
                            "token_type_ids": tf.TensorShape([None]),
                        },
                        tf.TensorShape([None]),
                    ),
                )

        def get_dataset(self):
            self.dataset = self.dataset.apply(
                tf.data.experimental.assert_cardinality(len(self.features)))

            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]
