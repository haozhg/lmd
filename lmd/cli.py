"""Console script for lmd."""
import argparse
import logging
import math
import os
import random
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Optional, Union

import torch
import transformers
from datasets import DatasetDict, load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level="INFO",
)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        "Language Model Decomposition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target", type=str, default="bert-base-uncased", help="target model in LMD"
    )
    parser.add_argument(
        "--basis",
        type=str,
        default="roberta-base",
        help="basis model in LMD, separated by comma",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="bert-base-uncased",
        help="tokenizer used for generating sequences (used by all models as text input)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="max_seq_length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="batch size for model inference",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="wikitext",
        help="The name of the dataset (corpus) to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset-config-name",
        type=str,
        default="wikitext-2-v1",
        help="The configuration name of the dataset (corpus) to use (via the datasets library).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=1280,
        help="max train samples",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=128,
        help="max validation samples",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=128,
        help="max test samples",
    )
    parser.add_argument(
        "--preprocessing-num-workers",
        type=int,
        default=None,
        help="preprocessing_num_workers for datasets.map()",
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="if we overwrite cache for datasets.map()",
    )
    parser.add_argument(
        "--preprocess-dir",
        type=str,
        default="data/preprocess",
        help="data dir to save preprocessed datasets",
    )
    parser.add_argument(
        "--embedding-dir",
        type=str,
        default="data/embeddings",
        help="data dir to save embedding datasets",
    )
    args = parser.parse_args()
    return args


def log_few_samples(raw_datasets: DatasetDict, k: int = 1):
    for split, ds in raw_datasets.items():
        for index in random.sample(range(len(ds)), k):
            logger.info(f"Sample {index} of the {split} set: {ds[index]}.")


def sample_datasets_subset(
    raw_datasets: DatasetDict, keep: Dict[str, int]
) -> DatasetDict:
    for split, max_samples in keep.items():
        if max_samples is not None and split in raw_datasets.keys():
            max_samples = min(len(raw_datasets[split]), max_samples)
            raw_datasets[split] = raw_datasets[split].select(range(max_samples))
    return raw_datasets


def gen_sentences(
    raw_datasets: DatasetDict, data_args: argparse.Namespace
) -> DatasetDict:
    """_summary_
    Reference
    https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py#L450

    Args:
        raw_datasets (DatasetDict): _description_
        data_args (argparse.Namespace): _description_

    Returns:
        DatasetDict: _description_
    """

    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_name)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    logger.info(f"after tokenization:\n{tokenized_datasets=}")
    log_few_samples(tokenized_datasets)

    filename = os.path.join(
        data_args.preprocess_dir, data_args.tokenizer_name, "tokenized"
    )
    logger.info(f"save tokenized_datasets to {filename}")
    tokenized_datasets.save_to_disk(filename)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    grouped_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {max_seq_length}",
    )

    logger.info(f"after grouping text:\n{grouped_datasets=}")
    log_few_samples(grouped_datasets)

    filename = os.path.join(
        data_args.preprocess_dir, data_args.tokenizer_name, "tokenized_grouped"
    )
    logger.info(f"save grouped_datasets to {filename}")
    grouped_datasets.save_to_disk(filename)

    # reconstruct original text
    # https://huggingface.co/docs/datasets/v2.5.2/en/package_reference/main_classes#datasets.Dataset.map
    # `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False`
    def get_sequence_text(examples: Dict[str, List]) -> Dict[str, List]:
        # https://huggingface.co/docs/transformers/main_classes/tokenizer
        # https://huggingface.co/docs/transformers/v4.22.2/en/main_classes/tokenizer#transformers.BatchEncoding
        # type(examples) = dict
        # examples.keys() = dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        # examples['input_ids'] is tensor of size (batch_size, seq_length)
        return {
            "text": tokenizer.batch_decode(
                examples["input_ids"], skip_special_tokens=True
            )
        }

    column_names = tokenized_datasets["train"].column_names

    sequence_datasets = grouped_datasets.map(
        get_sequence_text,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Get sequence text",
    )

    logger.info(f"after get_sequence_text():\n{sequence_datasets=}")
    log_few_samples(sequence_datasets)

    sequence_datasets = sample_datasets_subset(
        sequence_datasets,
        {
            "train": data_args.max_train_samples,
            "validation": data_args.max_val_samples,
            "test": data_args.max_test_samples,
        },
    )

    logger.info(f"after selecting subset\nsequence_datasets: {sequence_datasets}")

    filename = os.path.join(
        data_args.preprocess_dir, data_args.tokenizer_name, "tokenized_grouped_sequence"
    )
    logger.info(f"save sequence_datasets to {filename}")
    sequence_datasets.save_to_disk(filename)

    return sequence_datasets


# https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Pooling.py#L84
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(
    token_embeddings: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    # (batch_size, seq_len, hidden_size)
    # (batch_size, hidden_size)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.mean(token_embeddings * input_mask_expanded, dim=1)


def gen_embeddings(
    model_name_or_path: str, raw_datasets: DatasetDict, data_args: argparse.Namespace
) -> DatasetDict:
    """_summary_

    Reference
    https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py#L434

    Args:
        model_name_or_path (str): _description_
        raw_datasets (DatasetDict): _description_
        data_args (argparse.Namespace): _description_

    Returns:
        DatasetDict: _description_
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)
    model.to(dev)

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Tokenize all sequences
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            padding="max_length",
            max_length=max_seq_length,
            truncation=True,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on every text in dataset",
    )

    logger.info(f"after tokenization:\n{tokenized_datasets=}")
    log_few_samples(tokenized_datasets)
    filename = os.path.join(data_args.embedding_dir, model_name_or_path, "tokenized")
    logger.info(f"save tokenized_datasets to {filename}")
    tokenized_datasets.save_to_disk(filename)

    # https://huggingface.co/docs/datasets/use_with_pytorch
    logger.info("Set datasets format as torch Tensor")
    tokenized_datasets.set_format("torch")
    log_few_samples(tokenized_datasets)

    # compute embeddings
    # https://huggingface.co/docs/datasets/v2.5.2/en/package_reference/main_classes#datasets.Dataset.map
    # `function(batch: Dict[str, List]) -> Dict[str, List]` if `batched=True` and `with_indices=False`
    def get_sequence_embedding(examples: Dict[str, List]) -> Dict[str, List]:
        # https://huggingface.co/docs/transformers/main_classes/tokenizer
        # https://huggingface.co/docs/transformers/v4.22.2/en/main_classes/tokenizer#transformers.BatchEncoding
        # type(examples) = dict
        # examples.keys() = dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
        # examples['input_ids'] is tensor of size (batch_size, seq_length)
        examples = transformers.BatchEncoding(examples)
        examples.to(model.device)
        outputs = model(**examples)
        sentence_embeddings = mean_pooling(
            outputs.last_hidden_state, examples["attention_mask"]
        )
        return {"embedding": sentence_embeddings.detach().cpu().numpy()}

    column_names = tokenized_datasets["train"].column_names

    embedding_datasets = tokenized_datasets.map(
        get_sequence_embedding,
        batched=True,
        batch_size=data_args.batch_size,
        num_proc=data_args.preprocessing_num_workers,
        # remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Get sequence embeddings",
    )

    logger.info(f"after computing embedding:\n{embedding_datasets=}")
    log_few_samples(embedding_datasets)

    filename = os.path.join(
        data_args.embedding_dir, model_name_or_path, "tokenized_grouped_embeddings"
    )
    logger.info(f"save embedding_datasets to {filename}")
    embedding_datasets.save_to_disk(
        os.path.join(
            data_args.embedding_dir, model_name_or_path, "tokenized_grouped_embeddings"
        )
    )

    return embedding_datasets


def main():
    """Console script for lmd."""
    args = parse_args()

    args.basis = args.basis.split(",")

    print("Arguments: " + str(args))

    # load dataset
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

    logger.info(f"after loading datasets:\nraw_datasets:{raw_datasets}")

    raw_datasets = sample_datasets_subset(
        raw_datasets,
        {
            "train": args.max_train_samples * 10,
            "validation": args.max_val_samples * 10,
            "test": args.max_test_samples * 10,
        },
    )

    logger.info(f"after selecting subset\nraw_datasets: {raw_datasets}")

    log_few_samples(raw_datasets)

    # first use bert tokenizer to generate sentences
    sequence_datasets = gen_sentences(raw_datasets, args)

    # then tokenize using model specific tokenizers
    # compute embedding
    logger.info(f"gen embeddings for target model_name={args.target}")
    target_embedding_datasets = gen_embeddings(args.target, sequence_datasets, args)

    basis_embeddings = {}
    for model_name in args.basis:
        logger.info(f"gen embeddings for {model_name=}")
        basis_embeddings[model_name] = gen_embeddings(
            model_name, sequence_datasets, args
        )

    # solve LMD
    # output results


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover