#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
from datasets import load_from_disk, disable_progress_bars, disable_caching

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

# jean-zay personal imports
import idr_torch

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    streaming: bool = field(
        default=None,
        metadata={"help": "Whether to use the dataset in streaming"},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    filter_metric: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "select metric for filtering out examples below or/and above the specified threshold of dataset"
                "can be 'sjr' or 'h-index' "
                "if None, no filtering is done"
            )
        },
    )
    filter_lower_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "select lower threshold for filtering out examples below the specified threshold of dataset" 
                "must be set when filter_metric is set"
            )
        },
    )
    filter_upper_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "select upper threshold for filtering out examples above the specified threshold of dataset"
                "must be set when filter_metric is set"
            )
        },
    )
    metric_path: Optional[str] = field(
        default=None,
        metadata={"help": "path to the metric file (because this script is made to be run in offline mode)"},
    )
    disable_caching: bool = field(
        default=False, 
        metadata={"help": "Disable caching of the datasets"}
    )
    wandb_group: Optional[str] = field(
        default=None, 
        metadata={"help": "Group name for Weights and Biases (distributed run = 1 group)"}
    )
    wandb_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Run Name for Weights and Biases"}
    )
    def __post_init__(self):
        if self.dataset_name is None :
            raise ValueError("Need either a dataset path")
        if self.filter_metric is not None and self.filter_metric not in ["sjr", "h-index","random"]:
            raise ValueError("filter_metric must be either 'random', 'sjr' or 'h-index'")
        if self.filter_metric is not None and (self.filter_lower_threshold is None or self.filter_upper_threshold is None):
            raise ValueError("filter_lower_threshold and filter_upper_threshold must be set when filter_metric is set")
        if (self.filter_lower_threshold is not None 
            and self.filter_upper_threshold is not None 
            and self.filter_lower_threshold >= self.filter_upper_threshold):
            raise ValueError("filter_lower_threshold must be strictly lower than filter_upper_threshold")
        

def main():
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # set local rank to training arguments
    training_args.local_rank = idr_torch.rank
    os.environ["OMP_NUM_THREADS"] = "8"
    # setup wandb env variables
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_PROJECT"] = "pretrain-med-data-qual"
    os.environ["WANDB_RUN_GROUP"] = data_args.wandb_group
    os.environ["WANDB_NAME"] = data_args.wandb_name
    
    # disable progress bars (for file logging) 
    disable_progress_bars()
    if data_args.disable_caching:
        disable_caching()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if idr_torch.rank == 0 :
        if training_args.should_log:
            # The default of training_args.log_level is passive, so we set log level at info here to have that default.
            transformers.utils.logging.set_verbosity_info()

        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Dataset loading : dataset must be pre-tokenized and already splitted and shuffled
    tokenized_datasets = load_from_disk(
        data_args.dataset_name
    )

    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # we filter according to text quality
    if data_args.filter_metric is not None:
        # must not be batched as we want to check each example
        def metric_filter(examples, metric, lower_threshold, upper_threshold):
            return lower_threshold <= examples[metric] <= upper_threshold
        # we do not filter the validation set
        tokenized_datasets["train"] = tokenized_datasets["train"].filter(
            metric_filter,
            fn_kwargs={
                "metric": data_args.filter_metric,
                "lower_threshold": data_args.filter_lower_threshold,
                "upper_threshold": data_args.filter_upper_threshold,
            },
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Filtering examples where {data_args.filter_lower_threshold} <= {data_args.filter_metric} <= {data_args.filter_upper_threshold}",
        )
    
    # Remove filtering metrics columns
    columns_to_keep = [
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "special_tokens_mask",
    ]
    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        tokenized_datasets["train"] = tokenized_datasets["train"].remove_columns(
            [c for c in tokenized_datasets["train"].column_names
             if c not in columns_to_keep]
        )
    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        tokenized_datasets["validation"] = tokenized_datasets["validation"].remove_columns(
            [c for c in tokenized_datasets["validation"].column_names 
             if c not in columns_to_keep]
        )

    # Streaming needed for large datasets because we encountered NCCL timeout error
    # but we stream after filtering because faster (and no error during filtering)
    if data_args.streaming:
        tokenized_datasets = datasets.IterableDatasetDict({
            "train":tokenized_datasets["train"].to_iterable_dataset(num_shards=64),
            "validation":tokenized_datasets["validation"].to_iterable_dataset(num_shards=64),
        })
    # Define the maximum sequence length
    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # text is pre tokenized
    def group_tokens_and_pad(examples):
        # Concatenate all texts in the batch
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # padding variable
        remainder = total_length % max_seq_length 
        # update total length
        total_length = (total_length // max_seq_length) * max_seq_length 
        # Add padding tokens and masks
        if remainder != 0:
            total_length += max_seq_length 
            concatenated_examples["input_ids"] += [tokenizer.pad_token_id] * (max_seq_length - remainder)
            concatenated_examples["token_type_ids"] += [0] * (max_seq_length - remainder)
            concatenated_examples["attention_mask"] += [0] * (max_seq_length - remainder)
            concatenated_examples["special_tokens_mask"] += [1] * (max_seq_length - remainder)
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            tokenized_datasets = tokenized_datasets.map(
                group_tokens_and_pad,
                batched=True,
                batch_size=1024,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {max_seq_length}",
            ) 
        else:
            tokenized_datasets = tokenized_datasets.map(
                group_tokens_and_pad,
                batched=True,
            )
    

    if training_args.do_train:
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None :
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            if not data_args.streaming : 
                train_dataset = train_dataset.select(range(max_train_samples))
            else : 
                train_dataset = train_dataset.take(max_train_samples)
    if training_args.do_eval:
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None :
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            if not data_args.streaming : 
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            else : 
                eval_dataset = eval_dataset.take(max_eval_samples)

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load(data_args.metric_path, cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return metric.compute(predictions=preds, references=labels)
        
    # Data collator
    # This one will take care of randomly masking the tokens.
    pad_to_multiple_of_8 = training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        if idr_torch.rank == 0:
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        if idr_torch.rank == 0:
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()