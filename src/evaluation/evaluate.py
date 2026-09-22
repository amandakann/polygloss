import collections
import logging
from typing import Any, cast

import editdistance
import glossing
import pandas as pd
import pytest
import regex as re

from data.model import boundary_pattern
from src.dataset.prepare_dataset import split_interleaved_segments

from ..util.type_utils import all_not_none
from .alignment_score import alignment_score

logger = logging.getLogger(__name__)

UNK_GLOSSES = frozenset({"UNK", glossing.IGT.UNK_TOKEN})
"""Gold gloss tokens marking a morpheme the annotator could not gloss, 
excluded from the glossing metrics. Segmentation scoring is unaffected.
"""


def _rejoin(word: str, morphemes: list[str], keep: list[int]) -> str:
    """Reassembles the kept morphemes of a word, preserving its boundary characters."""
    dividers = re.findall(boundary_pattern, word) + [""]
    return "".join(
        morphemes[j] + (dividers[j] if k < len(keep) - 1 else "")
        for k, j in enumerate(keep)
    )


def _drop_unk_words(predicted: str, reference: str) -> tuple[str, str]:
    """Drops each word whose reference gloss contains an UNK morpheme, from both sides.
    Used for the word-level metrics.
    """
    predicted_words = predicted.split()
    reference_words = reference.split()
    keep = [
        i
        for i, word in enumerate(reference_words)
        if not UNK_GLOSSES.intersection(re.split(boundary_pattern, word))
    ]
    return (
        " ".join(predicted_words[i] for i in keep if i < len(predicted_words)),
        " ".join(reference_words[i] for i in keep),
    )


def _drop_unk_morphemes(predicted: str, reference: str) -> tuple[str, str]:
    """Drops UNK reference morphemes, and the predicted morphemes in the same positions,
    keeping the rest of the word.
    Used for the morpheme-level metrics.
    """
    predicted_words = predicted.split()
    reference_words = reference.split()
    out_predicted, out_reference = [], []
    for i, reference_word in enumerate(reference_words):
        morphemes = re.split(boundary_pattern, reference_word)
        keep = [j for j, m in enumerate(morphemes) if m not in UNK_GLOSSES]
        if not keep:
            # Every morpheme was UNK, so the word drops out entirely
            continue
        out_reference.append(_rejoin(reference_word, morphemes, keep))
        if i < len(predicted_words):
            predicted_morphemes = re.split(boundary_pattern, predicted_words[i])
            out_predicted.append(
                _rejoin(
                    predicted_words[i],
                    predicted_morphemes,
                    [j for j in keep if j < len(predicted_morphemes)],
                )
            )
    return " ".join(out_predicted), " ".join(out_reference)


def _evaluate_glosses(gloss_predictions: pd.DataFrame) -> dict[str, Any]:
    """Computes glossing metrics, excluding reference tokens the annotator marked UNK.

    The word and morpheme levels need different filtering - a word with one unknown morpheme
    can't be scored as a word, but its other morphemes still can - so each level is computed
    over its own filtered view.
    """
    predicted = gloss_predictions["predicted"].tolist()
    reference = gloss_predictions["reference"].tolist()

    views = {
        "words": [_drop_unk_words(p, r) for p, r in zip(predicted, reference)],
        "morphemes": [_drop_unk_morphemes(p, r) for p, r in zip(predicted, reference)],
    }

    metrics: dict[str, Any] = {}
    for level, view in views.items():
        # An entirely-UNK reference leaves nothing to score, and the library rejects it
        scoreable = [(p, r) for p, r in view if r.strip()]
        if len(scoreable) < len(view):
            logger.warning(
                f"Excluded {len(view) - len(scoreable)}/{len(view)} rows from the {level} "
                f"metrics: every reference token was UNK"
            )
        if not scoreable:
            logger.error(f"No reference tokens remain for the {level} metrics")
            continue
        generations, references = (list(side) for side in zip(*scoreable))
        level_metrics = glossing.evaluate_glosses(generations, references)
        metrics[level] = level_metrics[level]
        if level == "morphemes":
            # Character error rate follows the morpheme view, which keeps the glossed parts
            # of partially unknown words and preserves their boundary characters
            metrics["characters"] = level_metrics["characters"]
    return metrics


def evaluate(predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate predictions using appropriate metrics for glossing/segmentation.

    - For gloss predictions, compute metrics such as BLEU and morpheme accuracy.
    - For segmentation predictions, compute metrics such as F1 score.

    If multiple languages are present, we report both overall metrics and metrics per language
    """
    assert {"glottocode", "predicted", "reference", "task"}.issubset(
        predictions.columns
    )
    metrics = {}
    for glottocode, df in predictions.groupby("glottocode"):
        print(f"Evaluating: {glottocode} with {len(df)} rows")
        metrics[glottocode] = _evaluate(df)
    metrics["all"] = _evaluate(predictions)
    return metrics


def _evaluate(predictions: pd.DataFrame):
    gloss_predictions = predictions[predictions["task"].isin(["s2g", "t2g"])]
    segmentation_predictions = predictions[predictions["task"] == "t2s"]

    # Eval for joint t2sg task!!
    if (predictions["task"] == "t2sg").any():
        assert len(gloss_predictions) == 0
        assert len(segmentation_predictions) == 0
        gloss_label = "\nGlosses: "  # Split on this label
        joint_preds = predictions[predictions["task"] == "t2sg"]
        ref_split = (
            joint_preds["reference"]
            .str.split(  # type:ignore
                gloss_label, n=1, expand=True, regex=False
            )
            .fillna("")
        )
        pred_split = (
            joint_preds["predicted"]
            .str.split(  # type:ignore
                gloss_label, n=1, expand=True, regex=False
            )
            .fillna("")
        )
        segmentation_predictions = joint_preds.copy()
        gloss_predictions = joint_preds.copy()
        segmentation_predictions["reference"] = ref_split[0]
        segmentation_predictions["predicted"] = pred_split[0]
        gloss_predictions["reference"] = ref_split[1]
        gloss_predictions["predicted"] = pred_split[1]
    elif (predictions["task"] == "t2sg_interleaved").any():
        assert len(gloss_predictions) == 0
        assert len(segmentation_predictions) == 0
        joint_preds = cast(
            pd.DataFrame, predictions[predictions["task"] == "t2sg_interleaved"]
        )
        pred_split = (
            joint_preds["predicted"].apply(split_interleaved_segments).apply(pd.Series)
        )
        ref_split = (
            joint_preds["reference"].apply(split_interleaved_segments).apply(pd.Series)
        )
        segmentation_predictions = joint_preds.copy()
        gloss_predictions = joint_preds.copy()
        segmentation_predictions["reference"] = ref_split[0]
        segmentation_predictions["predicted"] = pred_split[0]
        gloss_predictions["reference"] = ref_split[1]
        gloss_predictions["predicted"] = pred_split[1]

    metrics: dict[str, dict | float] = {}

    if len(gloss_predictions) > 0:
        assert all_not_none(gloss_predictions["reference"].tolist())
        metrics["glossing"] = _evaluate_glosses(gloss_predictions)

    if len(segmentation_predictions) > 0:
        # Average metrics over examples

        segmentation_metrics = collections.defaultdict(float)
        for _, row in segmentation_predictions.iterrows():
            for k, v in _evaluate_segmentation_example(
                row["predicted"],  # type:ignore
                row["reference"],  # type:ignore
            ).items():
                segmentation_metrics[k] += v
        segmentation_metrics = {
            k: v / len(segmentation_predictions)
            for k, v in segmentation_metrics.items()
        }
        metrics["segmentation"] = segmentation_metrics

    if len(gloss_predictions) > 0 and len(segmentation_predictions) > 0:
        # We have both, let's calculate alignment

        assert len(gloss_predictions) == len(segmentation_predictions), (
            f"Must have same number of glossing and segmentation predictions. Got {len(gloss_predictions)} glosses and {len(segmentation_predictions)} segmentations."
        )
        joint_predictions = gloss_predictions.merge(
            segmentation_predictions, on="id", suffixes=("_glosses", "_segmentations")
        )
        metrics["alignment"] = alignment_score(
            [
                (s, g)
                for s, g in zip(
                    joint_predictions["predicted_segmentations"].tolist(),
                    joint_predictions["predicted_glosses"].tolist(),
                )
            ],
            should_log=True,
        )

    return metrics


def _evaluate_segmentation_example(generation: str, label: str):
    assert label is not None

    predicted_words = generation.split()
    label_words = label.split()

    predicted_morphemes = [re.split(boundary_pattern, word) for word in predicted_words]
    label_morphemes = [re.split(boundary_pattern, word) for word in label_words]

    # Compute modified f1
    total_predicted_morphs = sum(len(w) for w in predicted_morphemes)
    total_label_morphs = sum(len(w) for w in label_morphemes)
    total_overlapping = sum(
        _intersect_size(pred, label)
        for pred, label in zip(predicted_morphemes, label_morphemes)
    )
    precision = (
        total_overlapping / total_predicted_morphs if total_predicted_morphs > 0 else 0
    )
    recall = total_overlapping / total_label_morphs
    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    edit_dist = editdistance.eval(" ".join(predicted_words), " ".join(label_words))

    return {
        "accuracy": int(predicted_words == label_words),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "edit_distance": edit_dist,
    }


def _intersect_size(l1: list[str], l2: list[str]):
    """Computes the size of the intersection, allowing duplicate items, of two lists.
    i.e., if one list has the same element twice and the other has it once, we count this as 1
    """
    c1 = collections.Counter(l1)
    c2 = collections.Counter(l2)
    return sum(min(c1[k], c2[k]) for k in set(l1).intersection(set(l2)))


def test_evaluate_segmentation_example():
    metrics = _evaluate_segmentation_example(
        generation="t-he cat-s are run-ing",
        label="the cat-s are runn-ing",
    )
    assert metrics["accuracy"] == 0
    assert metrics["precision"] == 4 / 7
    assert metrics["recall"] == 4 / 6
    assert metrics["f1"] == pytest.approx(2 / ((7 / 4) + (6 / 4)))
