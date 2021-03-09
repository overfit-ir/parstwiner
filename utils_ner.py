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
    PreTrainedModel,
    Trainer,
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
    from torch.utils.data.sampler import RandomSampler
    import dataclasses
    from torch.utils.data.dataloader import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data.sampler import RandomSampler
    from typing import List, Union, Dict
    import numpy as np
    # from transformers.training_args import is_tpu_available
    # from transformers.trainer import get_tpu_sampler

    class MultitaskModel(PreTrainedModel):

        def __init__(self, encoder, taskmodels_dict):
            """
            Setting MultitaskModel up as a PretrainedModel allows us
            to take better advantage of Trainer features
            """
            super().__init__(PretrainedConfig())

            self.encoder = encoder
            self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

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
                model.to('cuda')
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
            if model_class_name.startswith("Bert"):
                return "bert"
            elif model_class_name.startswith("Roberta"):
                return "roberta"
            elif model_class_name.startswith("Albert"):
                return "albert"
            else:
                raise KeyError(f"Add support for new model {model_class_name}")

        def forward(self, task_name, **kwargs):
            return self.taskmodels_dict[task_name](**kwargs)

    class StrIgnoreDevice(str):
        """
        This is a hack. The Trainer is going call .to(device) on every input
        value, but we need to pass in an additional `task_name` string.
        This prevents it from throwing an error
        """

        def to(self, device):
            return self

    class DataLoaderWithTaskname:
        """
        Wrapper around a DataLoader to also yield a task name
        """

        def __init__(self, task_name, data_loader):
            self.task_name = task_name
            self.data_loader = data_loader

            self.batch_size = data_loader.batch_size
            self.dataset = data_loader.dataset

        def __len__(self):
            return len(self.data_loader)

        def __iter__(self):
            for batch in self.data_loader:
                batch["task_name"] = StrIgnoreDevice(self.task_name)
                yield batch

    class MultitaskDataloader:
        """
        Data loader that combines and samples from multiple single-task
        data loaders.
        """

        def __init__(self, dataloader_dict):
            self.dataloader_dict = dataloader_dict
            self.num_batches_dict = {
                task_name: len(dataloader)
                for task_name, dataloader in self.dataloader_dict.items()
            }
            self.task_name_list = list(self.dataloader_dict)
            self.dataset = [None] * sum(
                len(dataloader.dataset)
                for dataloader in self.dataloader_dict.values()
            )
            self.batch_size = 8

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
            for task_choice in task_choice_list:
                task_name = self.task_name_list[task_choice]
                yield next(dataloader_iter_dict[task_name])

    class MultitaskTrainer(Trainer):

        def get_single_train_dataloader(self, task_name, train_dataset):
            """
            Create a single-task data loader that also yields task names
            """
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
            # if is_tpu_available():
            #     train_sampler = get_tpu_sampler(train_dataset)
            else:
                train_sampler = (
                    RandomSampler(train_dataset)
                    if self.args.local_rank == -1
                    else DistributedSampler(train_dataset)
                )

            data_loader = DataLoaderWithTaskname(
                task_name=task_name,
                data_loader=DataLoader(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    sampler=train_sampler,
                    collate_fn=self.data_collator,
                ),
            )

            # if is_tpu_available():
            #     data_loader = pl.ParallelLoader(
            #         data_loader, [self.args.device]
            #     ).per_device_loader(self.args.device)

            return data_loader

        def get_single_eval_dataloader(self, task_name, eval_dataset):
            """
            Create a single-task data loader that also yields task names
            """
            if self.eval_dataset is None:
                raise ValueError(
                    "Trainer: evaluation requires a eval_dataset.")
            # if is_tpu_available():
            #     train_sampler = get_tpu_sampler(train_dataset)
            else:
                eval_sampler = (
                    RandomSampler(eval_dataset)
                    if self.args.local_rank == -1
                    else DistributedSampler(eval_dataset)
                )

            data_loader = DataLoaderWithTaskname(
                task_name=task_name,
                data_loader=DataLoader(
                    eval_dataset,
                    batch_size=self.args.train_batch_size,
                    sampler=eval_sampler,
                    collate_fn=self.data_collator,
                ),
            )

            # if is_tpu_available():
            #     data_loader = pl.ParallelLoader(
            #         data_loader, [self.args.device]
            #     ).per_device_loader(self.args.device)

            return data_loader

        def get_train_dataloader(self):
            """
            Returns a MultitaskDataloader, which is not actually a Dataloader
            but an iterable that returns a generator that samples from each
            task Dataloader
            """
            return MultitaskDataloader({
                task_name: self.get_single_train_dataloader(
                    task_name, task_dataset)
                for task_name, task_dataset in self.train_dataset.items()
            })

        def get_eval_dataloader(self, _):
            """
            Returns a MultitaskDataloader, which is not actually a Dataloader
            but an iterable that returns a generator that samples from each
            task Dataloader
            """
            if self.eval_dataset is None:
                raise ValueError(
                    "Trainer: evaluation requires a eval_dataset.")
            # if is_tpu_available():
            #     train_sampler = get_tpu_sampler(train_dataset)
            else:
                eval_sampler = (
                    RandomSampler(self.eval_dataset)
                    if self.args.local_rank == -1
                    else DistributedSampler(self.eval_dataset)
                )
            return DataLoaderWithTaskname(
                task_name='twitter',
                data_loader=DataLoader(
                    self.eval_dataset,
                    batch_size=self.args.train_batch_size,
                    sampler=eval_sampler,
                    collate_fn=self.data_collator,
                ),
            )

        def get_test_dataloader(self, test_dataset):
            """
            Returns a MultitaskDataloader, which is not actually a Dataloader
            but an iterable that returns a generator that samples from each
            task Dataloader
            """
            if test_dataset is None:
                raise ValueError(
                    "Trainer: evaluation requires a eval_dataset.")
            # if is_tpu_available():
            #     train_sampler = get_tpu_sampler(train_dataset)
            else:
                test_sampler = (
                    RandomSampler(test_dataset)
                    if self.args.local_rank == -1
                    else DistributedSampler(test_dataset)
                )
            return DataLoaderWithTaskname(
                task_name='twitter',
                data_loader=DataLoader(
                    test_dataset,
                    batch_size=self.args.train_batch_size,
                    sampler=test_sampler,
                    collate_fn=self.data_collator,
                ),
            )

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
