"""Exposes `prepare_dataset`.
Creates a dataset of examples designed for either a seq2seq or decoder-only causal LM (specifically Qwen 3).

- Examples will be tokenized and have:
    - separate input and label sequences (seq2seq)
    - combined prompt+completion sequences, with prompt masked in loss calculation (decoder-only causal LM)
- Which kind of examples are created is determined by the options in `experiment_config.py`
"""

import logging
import os
import typing
from pathlib import Path
from string import Template
from typing import Literal, cast

import datasets
import regex as re
from glossing.igt import gloss_string_to_word_glosses
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from data.model import boundary_pattern
from src.config.experiment_config import MODEL_TYPE, NO_TRANSLATION
from src.distributed import DistributedParameters
from src.train import ExperimentConfig
from src.util.collator import FlexibleCollatorWithPadding, FlexibleSeq2SeqCollator

logger = logging.getLogger(__name__)

supported_decoder_models = ["qwen3", "cohere"]

InputKey = typing.Literal["transcription", "segmentation"]
OutputKey = typing.Literal["segmentation", "glosses"]

def _load_local_dataset(path: str) -> datasets.DatasetDict:
    """Loads a local dataset from CSV files in split subdirectories (train/, dev/, test/)."""
    path = Path(path)
    
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {path}")
    
    data_files = {}
    for split_name in ["train", "dev", "test"]:
        split_dir = path / split_name
        if split_dir.is_dir():
            csv_files = sorted(split_dir.glob("*.csv"))
            if csv_files:
                data_files[split_name] = [str(f) for f in csv_files]
            else:
                logger.warning(f"No CSV files in {split_dir}")
    
    if not data_files:
        raise ValueError(f"No CSV files found in train/, dev/, or test/ subdirectories under {path}")
    
    return datasets.load_dataset("csv", data_files=data_files)


def _check_translation_condition(dataset: datasets.DatasetDict, config: ExperimentConfig):
    """Verifies the columns for the active translation condition exist and are populated.

    Without this, a misspelled or empty condition column falls through to the "no translation"
    fallback for every row, silently producing a duplicate of the base condition.
    """
    if not config.translation_condition or config.translation_condition == NO_TRANSLATION:
        return
    col = f"translation_{config.translation_condition}"
    metalang_col = f"metalanguage_{config.translation_condition}"
    for split in dataset:
        if col not in dataset[split].column_names:
            raise ValueError(
                f"Column {col!r} (translation_condition={config.translation_condition!r}) "
                f"not found in split {split!r}. Available: {dataset[split].column_names}"
            )
        if metalang_col not in dataset[split].column_names:
            raise ValueError(
                f"Column {metalang_col!r} not found in split {split!r}. Without it, prompts "
                f"would read 'Translation in an unknown language:'. "
                f"Available: {dataset[split].column_names}"
            )
        n_rows = dataset[split].num_rows
        n_filled = sum(1 for t in dataset[split][col] if t and t.strip())
        logger.info(f"Split '{split}': {n_filled}/{n_rows} rows have {col}")
        if n_filled == 0:
            raise ValueError(f"Column {col!r} is empty for every row in split {split!r}")
        if n_filled < n_rows:
            # Rows with no translation fall back to "Translation in English: None", so uneven
            # coverage across conditions confounds the comparison between them
            logger.warning(
                f"Column {col!r} is missing for {n_rows - n_filled}/{n_rows} rows in split "
                f"{split!r}; those rows get the no-translation prompt. Conditions are only "
                f"comparable if coverage matches across them."
            )


def create_dataset(
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
) -> datasets.DatasetDict:
    """Creates dataset for training/finetuning

    Args:
        tokenizer (transformers.AutoTokenizer): The pretrained tokenizer
        config (ExperimentConfig): The experiment configuration
    """
    # Load dataset from local CSV or HuggingFace Hub
    if config.local_dataset_path:
        dataset = _load_local_dataset(config.local_dataset_path)
    else:
        dataset = datasets.load_dataset(config.dataset_key)

    dataset = cast(datasets.DatasetDict, dataset)
    dataset = _filter(dataset, config.glottocode)
    _check_translation_condition(dataset, config)
    inputs_dataset = datasets.DatasetDict()

    if "glosslm" in config.pretrained_model:
        logger.warning("Detected GlossLM base model, using GlossLM prompt.")

    # Make examples with different input/output combos
    templates = {
        key: _load(key)
        for key in [
            "glosslm.t2g",
            "polygloss.multitask.t2g",
            "polygloss.multitask.t2s",
            "polygloss.multitask.s2g",
            "polygloss.concat.t2sg",
            "polygloss.interleaved.t2sg",
        ]
    }
    for split in dataset:
        if split == "train" and config.mode == "predict":
            continue
        examples = []
        skipped = 0
        for row in tqdm(dataset[split], f"Creating examples for {split}"):
            row = typing.cast(typing.Mapping, row)
            fields = _prepare_prompt_fields(row, config)

            if config.task_format == "gloss-only":
                if "glosslm" in config.pretrained_model:
                    raise NotImplementedError(
                        "GlossLM does not support the `gloss-only` format"
                    )
                prompt = templates["polygloss.multitask.t2g"].substitute(fields)
                examples.append(
                    _create_example(
                        prompt, fields["glosses"], "t2g", row, config.model_type
                    )
                )
                if split != "test":
                    prompt = templates["polygloss.multitask.s2g"].substitute(fields)
                    examples.append(
                        _create_example(
                            prompt, fields["glosses"], "s2g", row, config.model_type
                        )
                    )
            elif config.task_format == "segment-only":
                if "glosslm" in config.pretrained_model:
                    raise NotImplementedError(
                        "GlossLM does not support the `segment-only` format"
                    )
                if row["segmentation"]:
                    prompt = templates["polygloss.multitask.t2s"].substitute(fields)
                    examples.append(
                        _create_example(
                            prompt,
                            fields["segmentation"],
                            "t2s",
                            row,
                            config.model_type,
                        )
                    )
            elif config.task_format == "multitask":
                # Create transcription -> glosses and transcription -> segmentation for both splits
                # Create segmentation -> glosses for the train split ONLY
                if "glosslm" in config.pretrained_model:
                    prompt = templates["glosslm.t2g"].substitute(fields)
                else:
                    prompt = templates["polygloss.multitask.t2g"].substitute(fields)
                examples.append(
                    _create_example(
                        prompt, fields["glosses"], "t2g", row, config.model_type
                    )
                )
                if row["segmentation"]:
                    if "glosslm" not in config.pretrained_model:
                        prompt = templates["polygloss.multitask.t2s"].substitute(fields)
                        examples.append(
                            _create_example(
                                prompt,
                                fields["segmentation"],
                                "t2s",
                                row,
                                config.model_type,
                            )
                        )
                    if split != "test":
                        prompt = templates["polygloss.multitask.s2g"].substitute(fields)
                        examples.append(
                            _create_example(
                                prompt, fields["glosses"], "s2g", row, config.model_type
                            )
                        )
            elif config.task_format == "concatenated":
                # Create transcription -> glosses,
                #        segmentation -> glosses,
                #        transcription -> [segmentation, glosses] if possible
                if "glosslm" in config.pretrained_model:
                    raise NotImplementedError(
                        "GlossLM does not support the `concatenated` format"
                    )
                if config.mode == "grpo":
                    # GRPO doesn't need labels
                    prompt = templates["polygloss.concat.t2sg"].substitute(fields)
                    examples.append(
                        _create_example(prompt, None, "t2sg", row, config.model_type)
                    )
                    continue
                if fields["segmentation"]:
                    prompt = templates["polygloss.concat.t2sg"].substitute(fields)
                    label = f"{fields['segmentation']}\nGlosses: {fields['glosses']}"
                    examples.append(
                        _create_example(prompt, label, "t2sg", row, config.model_type)
                    )
                if split != "test":
                    prompt = templates["polygloss.multitask.t2g"].substitute(fields)
                    examples.append(
                        _create_example(
                            prompt, fields["glosses"], "t2g", row, config.model_type
                        )
                    )
                    if fields["segmentation"]:
                        prompt = templates["polygloss.multitask.s2g"].substitute(fields)
                        examples.append(
                            _create_example(
                                prompt, fields["glosses"], "s2g", row, config.model_type
                            )
                        )
            elif config.task_format == "interleaved":
                # Create transcription -> glosses,
                #        segmentation -> glosses,
                #        transcription -> interleaved[segmentation, glosses] if possible
                if "glosslm" in config.pretrained_model:
                    raise NotImplementedError(
                        "GlossLM does not support the `interleaved` format"
                    )
                if fields["segmentation"]:
                    prompt = templates["polygloss.interleaved.t2sg"].substitute(fields)

                    # Make the interleaved label
                    try:
                        label = create_interleaved_segments(
                            row["id"], fields["segmentation"], fields["glosses"]
                        )
                        examples.append(
                            _create_example(
                                prompt,
                                label,
                                "t2sg_interleaved",
                                row,
                                config.model_type,
                            )
                        )
                    except ValueError as e:
                        if split == "test":
                            logger.warning(e)
                        skipped += 1

                if split != "test":
                    prompt = templates["polygloss.multitask.t2g"].substitute(fields)
                    examples.append(
                        _create_example(
                            prompt, fields["glosses"], "t2g", row, config.model_type
                        )
                    )
                    if fields["segmentation"]:
                        prompt = templates["polygloss.multitask.s2g"].substitute(fields)
                        examples.append(
                            _create_example(
                                prompt, fields["glosses"], "s2g", row, config.model_type
                            )
                        )
            else:
                raise ValueError(f"Illegal value for task format: {config.task_format}")
        logger.info(f"Skipped {skipped} for split {split}")
        inputs_dataset[split] = datasets.Dataset.from_list(examples)

    model_config = AutoConfig.from_pretrained(config.pretrained_model)
    if model_config.is_encoder_decoder:
        _make_tokenizer = _make_seq2seq_tokenizer
    elif model_config.model_type in supported_decoder_models:
        # for now, only Qwen 3 is supported (uses chat template)
        _make_tokenizer = _make_causal_tokenizer_with_chat_template
    else:
        raise ValueError(
            f"Unknown or unsupported model_type from pretrained config: {model_config.model_type!r}. "
            f"Supported decoder-only model types: {supported_decoder_models}."
        )

    # Create prompts and tokenize
    return inputs_dataset.map(
        _make_tokenizer(tokenizer, max_length=config.max_tokens),
        batched=True,
        remove_columns=["input", "label"],
        desc="Tokenizing",
    )


def create_dataloader(
    dataset: datasets.Dataset,
    split: Literal["train", "dev", "test"],
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
    distributed_parameters: DistributedParameters,
):
    """Creates a dataloader for a dataset"""
    if config.model_type == "seq2seq":
        collator = FlexibleSeq2SeqCollator(tokenizer=tokenizer, label_pad_token_id=-100)
    elif config.model_type == "decoder":
        collator = FlexibleCollatorWithPadding(
            tokenizer=tokenizer,
            label_pad_token_id=-100,
            mlm=False,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
    if "SLURM_CPUS_PER_TASK" in os.environ:
        num_workers = int(os.environ["SLURM_CPUS_PER_TASK"])
    else:
        num_workers = 0
    if distributed_parameters["distributed"] and split == "train":
        sampler = DistributedSampler(
            dataset,  # type:ignore
            shuffle=True,
            num_replicas=distributed_parameters["world_size"],
            rank=distributed_parameters["rank"],
            drop_last=True,
        )
    else:
        sampler = (
            RandomSampler(dataset) if split == "train" else SequentialSampler(dataset)
        )
    return DataLoader(
        dataset,  # type:ignore
        batch_size=config.batch_size,
        collate_fn=collator,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=distributed_parameters["distributed"] and split == "train",
    )


def _filter(dataset: datasets.DatasetDict, glottocode: str | None):
    """Filter down to the relevant examples (depending on pretraining vs finetuning)"""
    dataset = dataset.filter(
        lambda x: x["transcription"] is not None and x["glosses"] is not None
    )
    new_dataset = datasets.DatasetDict()

    # Select the appropriate splits (ID or OOD)
    if glottocode is not None:
        print(f"Filtering to {glottocode=}")
        dataset = dataset.filter(lambda row: row["glottocode"] == glottocode)
        if dataset["test"].num_rows == 0:
            raise ValueError(f"Dataset does not contain glottocode {glottocode}!")
        new_dataset = dataset
    else:
        print("Re-splitting dataset for pretraining")
        # We must be pretraining
        # Instead of language-specific eval sets, let's use all of the pretraining and ID eval data and make iid splits
        pretraining_data = datasets.concatenate_datasets(
            [dataset["train"], dataset["dev"]]
        )
        pretraining_data = pretraining_data.train_test_split(test_size=0.1, seed=0)
        new_dataset["train"] = pretraining_data["train"]
        new_dataset["dev"] = pretraining_data["test"]
        new_dataset["test"] = dataset["test"]
    return new_dataset


def _make_seq2seq_tokenizer(tokenizer: PreTrainedTokenizerBase, max_length: int):
    """Tokenizer function for seq2seq models"""

    def _tokenize(batch):
        nonlocal tokenizer, max_length
        targets = batch.get("label")
        if any([t is None for t in targets]):
            targets = None
        model_inputs = tokenizer(
            batch["input"],
            text_target=targets,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        return model_inputs

    return _tokenize


def _make_causal_tokenizer_with_chat_template(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    use_thinking: bool = False,
):
    """Tokenizer function for decoder-only causal LMs using chat templates, specifically Qwen 3.
    Args:
        tokenizer: The tokenizer instance
        max_length: Maximum sequence length
        use_thinking: Whether to enable Qwen 3's thinking mode (default: False)
    """

    def _tokenize(batch):
        nonlocal tokenizer, max_length, use_thinking

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        user_and_assistant_turns = []
        prompt_lengths = []  # Track where assistant response starts for loss masking

        for prompt, label in zip(batch["input"], batch["label"]):
            if label is not None:
                # Apply chat template
                user_and_assistant_turns.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": label},
                        ],
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=use_thinking,
                    )
                )
            else:
                user_and_assistant_turns.append(
                    tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": prompt},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=use_thinking,
                    )
                )
            prompt_lengths.append(
                len(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=True,
                        add_generation_prompt=True,  # True to get the assistant start token
                        enable_thinking=use_thinking,
                    )["input_ids"]  # type:ignore
                )
            )

        # Tokenize the full conversations
        model_inputs = tokenizer(
            user_and_assistant_turns,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        model_inputs["prompt_lengths"] = prompt_lengths
        return model_inputs

    return _tokenize


def _prepare_prompt_fields(row: typing.Mapping, config: ExperimentConfig):
    """Given a row from the dataset, prepares the fields for the prompts"""
    transcription = " ".join((row["transcription"]).split())
    glosses = " ".join((row["glosses"]).split())
    if row["segmentation"] and len(row["segmentation"].strip()) > 0:
        segmentation = " ".join((row["segmentation"]).split())
    else:
        segmentation = None
    if config.language_mask:
        lang = config.language_mask
    else:
        lang = (
            "an unknown language"
            if row["language"] == "" or not row["language"]
            else row["language"]
        )
    if config.translation_condition == NO_TRANSLATION:
        translation, metalang = None, None
    elif config.translation_condition:
        translation = row.get(f"translation_{config.translation_condition}")
        metalang = row.get(f"metalanguage_{config.translation_condition}")
    else:
        translation = row["translation"]
        metalang = row["metalanguage"]

    if translation and len(translation.strip()) > 0 and translation != "Unknown":
        translation = " ".join(translation.split())
        metalang = metalang or "an unknown language"
    else:
        translation = "None"
        metalang = "English"
    return {
        "transcription": transcription,
        "glosses": glosses,
        "segmentation": segmentation,
        "translation": translation,
        "lang": lang,
        "metalang": metalang,
    }


def _load(prompt_key: str):
    """Loads a prompt and returns a template"""
    prompt_path = Path(__file__).parent / (prompt_key + ".prompt")
    with open(prompt_path, "r") as prompt_file:
        prompt_text = prompt_file.read()
    return Template(prompt_text)


def create_interleaved_segments(id: str, segmentation_str: str, gloss_string: str):
    """Given a plain-text segmentation and gloss string, creates an interleaved string.
    For example, given:
        Segments: "the cat-s run"
        Glosses: "DET cat-PL run.SG"
    We would return
        "DET(the) cat(cat)-PL(s) run.SG(run)"
    """
    label = []
    segment_words = gloss_string_to_word_glosses(segmentation_str)
    gloss_words = gloss_string_to_word_glosses(gloss_string)
    if len(segment_words) != len(gloss_words):
        raise ValueError(
            f"Word mismatch! ID: {id}\nS: {segmentation_str}\nG: {gloss_string}"
        )
    for s_word, g_word in zip(segment_words, gloss_words):
        segments = re.split(boundary_pattern, s_word)
        glosses = re.split(boundary_pattern, g_word)
        if len(segments) != len(glosses):
            raise ValueError(
                f"Morpheme mismatch! ID: {id}\nS: {segmentation_str}\nG: {gloss_string}"
            )
        dividers = re.findall(boundary_pattern, g_word) + [""]
        word = "".join([f"{g}({s}){d}" for g, s, d in zip(glosses, segments, dividers)])
        label.append(word)
    label = " ".join(label)
    return label


def split_interleaved_segments(interleaved: str):
    """Given an interleaved string, splits into separate segment and gloss strings.
    For example, given:
        "DET(the) cat(cat)-PL(s) run.SG(run)"
    We would return
        Segments: "the cat-s run"
        Glosses: "DET cat-PL run.SG"
    """
    words = interleaved.split()
    segmentation_words = []
    gloss_words = []
    for word in words:
        interleaved_segments = re.split(boundary_pattern, word)
        dividers = re.findall(boundary_pattern, word) + [""]
        word_segments = []
        word_glosses = []
        for int_segment in interleaved_segments:
            match = re.match(r"(.*)\((.*)\)", int_segment)
            if match:
                word_glosses.append(match.group(1))
                word_segments.append(match.group(2))
            else:
                # If we can't match the pattern, must be a bad prediction
                # Just add as a gloss
                word_glosses.append(int_segment)
                word_segments.append("")
        segmentation_words.append(
            "".join([f"{s}{d}" for s, d in zip(word_segments, dividers)])
        )
        gloss_words.append("".join([f"{g}{d}" for g, d in zip(word_glosses, dividers)]))
    segmentation = " ".join(segmentation_words)
    glosses = " ".join(gloss_words)
    return segmentation, glosses


def _create_example(
    prompt: str,
    label: str | None,
    task_key: str,
    row: typing.Mapping,
    model_type: MODEL_TYPE,
):
    if model_type in ["seq2seq", "decoder"]:
        return {
            "input": prompt,
            "label": label,
            "task": task_key,
            "id": row["id"],
            "glottocode": row["glottocode"],
        }
    else:
        raise NotImplementedError()
