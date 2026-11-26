"""JAX downstream fine-tuning for the Vietnamese suite."""

from __future__ import annotations

import ast
import hashlib
import math
import random
import re
import string
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from hamic_jax.model import encoder_hidden
from hamic_jax.norms import layer_norm
from hamic_jax.optim import clip_by_global_norm
from hamic_jax.spec import ModelPreset

__all__ = ["TASKS", "FinetuneConfig", "TaskData", "TaskResult", "TaskSpec", "build_tokenizer", "load_task", "run_task"]

IGNORE = -100
_VSFC = {"train": "default/train/0000.parquet", "validation": "default/validation/0000.parquet", "test": "default/test/0000.parquet"}


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    dataset: str
    revision: str
    config: str | None = None
    splits: dict[str, str] = field(default_factory=dict)
    fields: dict[str, str] = field(default_factory=dict)
    metric: str = "weighted_f1"
    parquet: dict[str, str] = field(default_factory=dict)


def _spec(name: str, kind: str, dataset: str, revision: str, splits: dict[str, str], fields: dict[str, str], **kwargs: Any) -> TaskSpec:
    return TaskSpec(name, kind, dataset, revision, splits=splits, fields=fields, **kwargs)


TASKS = {
    item.name: item
    for item in [
        _spec("phoner", "token", "hungphongtrn/PhoNer_Covid19", "7803a7d58c2d31a90a14e3776811f176a2e5878a", {"train": "train", "dev": "validation", "test": "test"}, {"tokens": "words", "label": "tags"}, metric="entity_f1"),
        _spec("vsmec", "sequence", "tridm/UIT-VSMEC", "6feda28cba29ea564c76a75b2c77f9af1fbc56e6", {"train": "train", "dev": "validation", "test": "test"}, {"text": "Sentence", "label": "Emotion"}),
        _spec("vihsd", "sequence", "ura-hcmut/UIT-ViHSD", "16c4f67cf509d4f9f36ca5b63c5503c7c8830557", {"train": "train", "dev": "validation", "test": "test"}, {"text": "free_text", "label": "label_id"}),
        _spec("vinli", "pair", "presencesw/vinli_4_label", "a885fd1bdeb00099f2215b601ddb0df255503cf1", {"train": "train", "dev": "dev", "test": "test"}, {"text": "sentence1", "text_b": "sentence2", "label": "gold_label"}),
        _spec("viquad2", "qa", "taidng/UIT-ViQuAD2.0", "406f09a45cc106a8f7b7fd0c25078883fe58cb1f", {"train": "train", "test": "validation"}, {}, metric="f1"),
        _spec("pos", "token", "universal-dependencies/universal_dependencies", "c3fed78df6b9f8cd27ed24cc7142e9eccf1e6703", {"train": "train", "dev": "dev", "test": "test"}, {"tokens": "tokens", "label": "upos"}, config="vi_vtb"),
        _spec("vsfc_sentiment", "sequence", "uitnlp/vietnamese_students_feedback", "2d76906157232f87e9883fa37ebfc91ffdee1f83", {"train": "train", "dev": "validation", "test": "test"}, {"text": "sentence", "label": "sentiment"}, parquet=_VSFC),
        _spec("vsfc_topic", "sequence", "uitnlp/vietnamese_students_feedback", "2d76906157232f87e9883fa37ebfc91ffdee1f83", {"train": "train", "dev": "validation", "test": "test"}, {"text": "sentence", "label": "topic"}, parquet=_VSFC),
        _spec("vtoc", "sequence", "tmnam20/ViGLUE", "ad7df00441b9e8666f107b12f6b3d3b3e16cb470", {"train": "train", "test": "validation"}, {"text": "sentence", "label": "label"}, config="vtoc"),
    ]
}


@dataclass
class TaskData:
    spec: TaskSpec
    splits: dict[str, list[dict[str, Any]]]
    labels: list[str]


def _bucket(key: str) -> float:
    digest = hashlib.sha256(f"hamic-dev-v1:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _carve_dev(rows: list[dict[str, Any]], key: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, dev = [], []
    for index, row in enumerate(rows):
        value = str(row.get(key, index)) if key else str(index)
        (dev if _bucket(value) < 0.1 else train).append(row)
    return train, dev


def _parse(value: Any) -> Any:
    return ast.literal_eval(value) if isinstance(value, str) and value[:1] in "[{" else value


def _load_split(spec: TaskSpec, split: str, cache_dir: str | None) -> Any:
    from datasets import load_dataset
    if spec.parquet:
        url = f"hf://datasets/{spec.dataset}@{spec.revision}/{spec.parquet[split]}"
        return load_dataset("parquet", data_files={split: url}, split=split, cache_dir=cache_dir)
    return load_dataset(spec.dataset, spec.config, split=split, revision=spec.revision, cache_dir=cache_dir)


def _label_names(feature: Any) -> list[str] | None:
    inner = getattr(feature, "feature", feature)
    return list(inner.names) if hasattr(inner, "names") else None


def _name(value: Any, names: list[str] | None) -> str:
    return names[int(value)] if names is not None and str(value).lstrip("-").isdigit() else str(value)


def _qa_row(row: dict[str, Any]) -> dict[str, Any]:
    answers = _parse(row.get("answers")) or {}
    impossible = str(row.get("is_impossible", "False")).lower() == "true"
    texts = [] if impossible else list(answers.get("text", []))
    starts = [] if impossible else [int(value) for value in answers.get("answer_start", [])]
    return {"id": str(row["id"]), "context": str(row["context"]), "question": str(row["question"]), "answers": [{"text": a, "start": b} for a, b in zip(texts, starts)]}


def load_task(spec: TaskSpec, cache_dir: str | None = None) -> TaskData:
    raw = {role: _load_split(spec, split, cache_dir) for role, split in spec.splits.items()}
    if spec.kind == "qa":
        splits = {role: [_qa_row(row) for row in rows] for role, rows in raw.items()}
        if "dev" not in splits:
            titles = {row["id"]: title for row, title in zip(splits["train"], raw["train"]["title"])}
            for row in splits["train"]:
                row["title"] = titles[row["id"]]
            splits["train"], splits["dev"] = _carve_dev(splits["train"], "title")
        return TaskData(spec, splits, [])
    label_field = spec.fields["label"]
    names = _label_names(raw["train"].features[label_field])
    splits = {}
    for role, rows in raw.items():
        if spec.kind == "token":
            splits[role] = [{"tokens": list(_parse(row[spec.fields["tokens"]])), "tags": [_name(tag, names) for tag in _parse(row[label_field])]} for row in rows]
        else:
            splits[role] = [{"text": str(row[spec.fields["text"]]), **({"text_b": str(row[spec.fields["text_b"]])} if "text_b" in spec.fields else {}), "label": _name(row[label_field], names)} for row in rows]
    if "dev" not in splits:
        splits["train"], splits["dev"] = _carve_dev(splits["train"], "")
    if spec.kind == "token":
        labels = sorted({tag for row in splits["train"] for tag in row["tags"]})
    else:
        labels = sorted({row["label"] for row in splits["train"]})
        index = {label: position for position, label in enumerate(labels)}
        for rows in splits.values():
            for row in rows:
                row["label"] = index.get(row["label"], -1)
    return TaskData(spec, splits, labels)

def build_tokenizer(tokenizer_file: str | Path, max_length: int = 512) -> Any:
    from transformers import AddedToken, PreTrainedTokenizerFast
    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_file), bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>", cls_token="<s>", sep_token="</s>",
        mask_token=AddedToken("<mask>", lstrip=True, rstrip=False, special=True),
        model_max_length=max_length, clean_up_tokenization_spaces=False,
    )


@dataclass(frozen=True)
class FinetuneConfig:
    learning_rates: tuple[float, ...] = (2e-5, 5e-5)
    seeds: tuple[int, ...] = (10, 42, 100)
    batch_size: int = 32
    max_length: int = 512
    max_epochs: int = 10
    patience: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    qa_stride: int = 128
    qa_max_answer_tokens: int = 64


@dataclass
class TaskResult:
    task: str
    metric: str
    learning_rate: float
    dev_by_rate: dict[float, float] = field(default_factory=dict)
    test_by_seed: dict[int, dict[str, float]] = field(default_factory=dict)
    seconds: float = 0.0


def _encode_classification(rows, tokenizer, max_length):
    texts = [row["text"] for row in rows]
    pairs = [row["text_b"] for row in rows] if rows and "text_b" in rows[0] else None
    encoded = tokenizer(texts, pairs, truncation=True, max_length=max_length, padding="max_length")
    return [{"input_ids": ids, "labels": row["label"]} for ids, row in zip(encoded["input_ids"], rows)]


def _encode_tokens(rows, tokenizer, labels, max_length):
    index = {label: position for position, label in enumerate(labels)}
    features = []
    for row in rows:
        starts, text = [], ""
        for position, token in enumerate(row["tokens"]):
            if position:
                text += " "
            starts.append(len(text))
            text += token
        encoded = tokenizer(text, truncation=True, max_length=max_length, return_offsets_mapping=True)
        word_at = {start: position for position, start in enumerate(starts)}
        label_ids, first = [], {}
        for position, (start, end) in enumerate(encoded["offset_mapping"]):
            while start < end and text[start] == " ":
                start += 1
            word = word_at.get(start) if end > start else None
            if word is not None and word not in first:
                first[word] = position
                label_ids.append(index.get(row["tags"][word], IGNORE))
            else:
                label_ids.append(IGNORE)
        features.append(
            {
                "input_ids": encoded["input_ids"],
                "labels": label_ids,
                "first_token": first,
            }
        )
    return features


def _encode_qa(rows, tokenizer, max_length, stride):
    encoded = tokenizer(
        [row["question"] for row in rows], [row["context"] for row in rows],
        truncation="only_second", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
    )
    features = []
    for index, ids in enumerate(encoded["input_ids"]):
        example = encoded["overflow_to_sample_mapping"][index]
        sequence_ids = encoded.sequence_ids(index)
        offsets = [offset if kind == 1 else None for offset, kind in zip(encoded["offset_mapping"][index], sequence_ids)]
        answers = rows[example]["answers"]
        start = end = 0
        if answers:
            answer_start = answers[0]["start"]
            answer_end = answer_start + len(answers[0]["text"])
            spans = {position: offset for position, offset in enumerate(offsets) if offset}
            positions = sorted(spans)
            if positions and spans[positions[0]][0] <= answer_start and spans[positions[-1]][1] >= answer_end:
                start = next(p for p in positions if spans[p][1] > answer_start)
                end = next(p for p in reversed(positions) if spans[p][0] < answer_end)
        features.append({"input_ids": ids, "start": start, "end": end, "example": example, "offsets": offsets})
    return features


def _pad(features, pad_id):
    length = max(len(feature["input_ids"]) for feature in features)
    ids = np.full((len(features), length), pad_id, dtype=np.int32)
    mask = np.zeros((len(features), length), dtype=np.bool_)
    for row, feature in enumerate(features):
        tokens = feature["input_ids"]
        ids[row, : len(tokens)] = tokens
        mask[row, : len(tokens)] = True
    batch = {"input_ids": ids, "attention_mask": mask}
    if "start" in features[0]:
        batch["start"] = np.asarray([feature["start"] for feature in features], np.int32)
        batch["end"] = np.asarray([feature["end"] for feature in features], np.int32)
    elif isinstance(features[0].get("labels"), list):
        labels = np.full((len(features), length), IGNORE, np.int32)
        for row, feature in enumerate(features):
            values = feature["labels"][:length]
            labels[row, : len(values)] = values
        batch["labels"] = labels
    else:
        batch["labels"] = np.asarray([feature["labels"] for feature in features], np.int32)
    return batch


def _init_head(hidden_size, outputs, key):
    """Initialise the oracle's ``ModernBertPredictionHead`` plus its classifier.

    ``AutoModelForSequenceClassification`` and its task siblings wrap the encoder in
    ``ModernBertPredictionHead`` (dense, GELU, LayerNorm) before a linear classifier.
    The head is re-initialised by ``from_pretrained``; it never reuses the MLM head.
    """
    dense_key, head_key = jax.random.split(key)
    return {
        "dense.weight": (
            jax.random.normal(dense_key, (hidden_size, hidden_size), jnp.float32) * 0.02
        ),
        "norm.weight": jnp.ones((hidden_size,), jnp.float32),
        "classifier.weight": (
            jax.random.normal(head_key, (outputs, hidden_size), jnp.float32) * 0.02
        ),
        "classifier.bias": jnp.zeros((outputs,), jnp.float32),
    }


def _task_logits(hidden, head, preset):
    """Apply ``ModernBertPredictionHead`` then the task classifier."""
    projected = hidden @ head["dense.weight"].T
    projected = jax.nn.gelu(projected, approximate=False)
    projected = layer_norm(projected, head["norm.weight"], eps=preset.norm_eps)
    return projected @ head["classifier.weight"].T + head["classifier.bias"]


def _decay_mask(encoder, head):
    """Match the oracle's two AdamW groups: no decay for vectors or normalization."""
    encoder_mask = {
        name: leaf.ndim >= 2 and "norm" not in name for name, leaf in encoder.items()
    }
    head_mask = {name: leaf.ndim >= 2 for name, leaf in head.items()}
    return (encoder_mask, head_mask)


def _loss(encoder, head, preset, batch, kind):
    hidden = encoder_hidden(encoder, preset, batch["input_ids"], batch["attention_mask"], document_mask=False)
    if kind in ("sequence", "pair"):
        logits = _task_logits(hidden[:, 0], head, preset)
        return optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), batch["labels"]).mean()
    if kind == "token":
        logits = _task_logits(hidden, head, preset)
        valid = batch["labels"] != IGNORE
        losses = optax.softmax_cross_entropy_with_integer_labels(logits.astype(jnp.float32), jnp.where(valid, batch["labels"], 0))
        return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(jnp.sum(valid), 1)
    logits = _task_logits(hidden, head, preset)
    start = optax.softmax_cross_entropy_with_integer_labels(logits[:, :, 0].astype(jnp.float32), batch["start"])
    end = optax.softmax_cross_entropy_with_integer_labels(logits[:, :, 1].astype(jnp.float32), batch["end"])
    return jnp.mean(start + end)


def _finetune_schedule(learning_rate, steps, warmup_ratio):
    total = max(1, steps)
    warmup = max(1, round(total * warmup_ratio))
    warm = optax.linear_schedule(
        init_value=learning_rate / warmup,
        end_value=learning_rate,
        transition_steps=max(1, warmup - 1),
    )
    if total <= warmup:
        return warm
    decay = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0.0,
        transition_steps=total - warmup,
    )
    return optax.join_schedules([warm, decay], [warmup])


def _train_fold(preset, encoder, features, labels, kind, config, learning_rate, seed, evaluate, pad_id):
    head = _init_head(preset.hidden_size, len(labels) if kind != "qa" else 2, jax.random.PRNGKey(seed))
    steps = max(1, math.ceil(len(features) / config.batch_size) * config.max_epochs)
    schedule = _finetune_schedule(learning_rate, steps, config.warmup_ratio)
    optimizer = optax.chain(
        clip_by_global_norm(config.grad_clip),
        optax.adamw(
            schedule,
            weight_decay=config.weight_decay,
            mask=_decay_mask(encoder, head),
        ),
    )
    state = optimizer.init((encoder, head))

    @jax.jit
    def step(params, opt_state, batch):
        value, grads = jax.value_and_grad(_loss)(params[0], params[1], preset, batch, kind)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, value

    order = list(range(len(features)))
    best_score, best_encoder, best_head, stale = -math.inf, encoder, head, 0
    for epoch in range(config.max_epochs):
        random.Random(seed * 1000 + epoch).shuffle(order)
        for start in range(0, len(order), config.batch_size):
            chunk = [features[index] for index in order[start : start + config.batch_size]]
            batch = {name: jnp.asarray(value) for name, value in _pad(chunk, pad_id).items()}
            (encoder, head), state, _ = step((encoder, head), state, batch)
        score = evaluate(encoder, head)
        if score > best_score:
            best_score, best_encoder, best_head, stale = score, encoder, head, 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    return best_encoder, best_head, best_score


def _predict(preset, encoder, head, features, kind, pad_id, batch_size):
    """Predict in the oracle's evaluation batches (``2 * train batch`` by default)."""
    if not features:
        return []
    outputs = []
    for start in range(0, len(features), batch_size):
        chunk = features[start : start + batch_size]
        batch = {name: jnp.asarray(value) for name, value in _pad(chunk, pad_id).items()}
        hidden = encoder_hidden(
            encoder,
            preset,
            batch["input_ids"],
            batch["attention_mask"],
            document_mask=False,
        )
        if kind in ("sequence", "pair"):
            logits = _task_logits(hidden[:, 0], head, preset)
            outputs.extend(np.asarray(jnp.argmax(logits, axis=-1)).tolist())
        elif kind == "token":
            logits = _task_logits(hidden, head, preset)
            outputs.append(np.asarray(jnp.argmax(logits, axis=-1)))
        else:
            logits = _task_logits(hidden, head, preset)
            outputs.append(
                (
                    np.asarray(logits[:, :, 0]),
                    np.asarray(logits[:, :, 1]),
                )
            )
    if kind in ("sequence", "pair"):
        return outputs
    if kind == "token":
        return np.concatenate(outputs, axis=0)
    return (
        np.concatenate([value[0] for value in outputs], axis=0),
        np.concatenate([value[1] for value in outputs], axis=0),
    )


def _classification_scores(predictions, labels):
    classes = sorted(set(labels) | set(predictions))
    support = Counter(labels)
    per_class = {}
    for item in classes:
        tp = sum(1 for p, y in zip(predictions, labels) if p == item and y == item)
        fp = sum(1 for p, y in zip(predictions, labels) if p == item and y != item)
        fn = sum(1 for p, y in zip(predictions, labels) if p != item and y == item)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[item] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = len(labels) or 1
    accuracy = sum(1 for p, y in zip(predictions, labels) if p == y) / total
    labelled = [item for item in classes if support[item]]
    macro = sum(per_class[item] for item in labelled) / (len(labelled) or 1)
    weighted = sum(per_class[item] * support[item] for item in labelled) / total
    return {"accuracy": round(100 * accuracy, 2), "macro_f1": round(100 * macro, 2), "weighted_f1": round(100 * weighted, 2)}


def _spans(tags):
    spans, start, kind = set(), -1, ""
    for index, tag in enumerate([*tags, "O"]):
        prefix, _, label = tag.partition("-")
        continues = prefix == "I" and kind == label and start >= 0
        if start >= 0 and not continues:
            spans.add((start, index, kind))
            start, kind = -1, ""
        if prefix in {"B", "I"} and not continues:
            start, kind = index, label
    return spans


def _entity_f1(predictions, labels):
    tp = fp = fn = 0
    for pred, gold in zip(predictions, labels):
        left, right = _spans(pred), _spans(gold)
        tp += len(left & right)
        fp += len(left - right)
        fn += len(right - left)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return round(100 * 2 * precision * recall / (precision + recall), 2) if precision + recall else 0.0


def _token_weighted_f1(predictions, labels):
    flat_p = [tag for row in predictions for tag in row]
    flat_y = [tag for row in labels for tag in row]
    names = sorted(set(flat_p) | set(flat_y))
    index = {name: position for position, name in enumerate(names)}
    return _classification_scores([index[tag] for tag in flat_p], [index[tag] for tag in flat_y])["weighted_f1"]


def _normalize(text):
    punctuation = re.compile(f"[{re.escape(string.punctuation)}“”‘’…–—]")
    return " ".join(punctuation.sub(" ", unicodedata.normalize("NFC", text).lower()).split())


def _squad_f1(prediction, gold):
    left, right = prediction.split(), gold.split()
    if not left or not right:
        return float(left == right)
    common = sum((Counter(left) & Counter(right)).values())
    if not common:
        return 0.0
    precision, recall = common / len(left), common / len(right)
    return 2 * precision * recall / (precision + recall)


def _qa_scores(predictions, answers):
    exact = f1 = 0.0
    for prediction, golds in zip(predictions, answers):
        candidates = [_normalize(gold) for gold in golds] or [""]
        value = _normalize(prediction)
        exact += max(float(value == candidate) for candidate in candidates)
        f1 += max(_squad_f1(value, candidate) for candidate in candidates)
    total = len(predictions) or 1
    return {"exact_match": round(100 * exact / total, 2), "f1": round(100 * f1 / total, 2)}


def _qa_answers(features, start_logits, end_logits, rows, max_answer_tokens):
    """Pick the best context span per example, with the oracle's null-answer rule."""
    best: dict[int, tuple[float, str]] = {}
    null: dict[int, float] = {}
    for index, feature in enumerate(features):
        offsets = feature["offsets"]
        example = feature["example"]
        start = start_logits[index]
        end = end_logits[index]
        null[example] = min(null.get(example, math.inf), float(start[0] + end[0]))
        context = [
            position for position, offset in enumerate(offsets) if offset is not None
        ]
        if not context:
            continue
        top_start = sorted(
            context, key=lambda position: -float(start[position])
        )[:20]
        top_end = sorted(
            context, key=lambda position: -float(end[position])
        )[:20]
        for first in top_start:
            for last in top_end:
                if last < first or last - first + 1 > max_answer_tokens:
                    continue
                score = float(start[first] + end[last])
                if score > best.get(example, (-math.inf, ""))[0]:
                    text = rows[example]["context"][offsets[first][0] : offsets[last][1]]
                    best[example] = (score, text)
    answers = []
    for example in range(len(rows)):
        span_score, text = best.get(example, (-math.inf, ""))
        answers.append(text if span_score > null.get(example, math.inf) else "")
    return answers


def run_task(preset: ModelPreset, encoder: dict[str, Any], data: TaskData, tokenizer: Any, config: FinetuneConfig | None = None, log: Callable[[str], None] = print) -> TaskResult:
    """Choose the learning rate on dev, then report test over every seed."""
    config = config or FinetuneConfig()
    spec = data.spec
    pad_id = int(tokenizer.pad_token_id)
    started = time.monotonic()
    if spec.kind in {"sequence", "pair"}:
        encoded = {role: _encode_classification(rows, tokenizer, config.max_length) for role, rows in data.splits.items()}
    elif spec.kind == "token":
        encoded = {role: _encode_tokens(rows, tokenizer, data.labels, config.max_length) for role, rows in data.splits.items()}
    else:
        encoded = {role: _encode_qa(rows, tokenizer, config.max_length, config.qa_stride) for role, rows in data.splits.items()}
    encoder = {
        name: jnp.asarray(value, dtype=jnp.float32)
        for name, value in encoder.items()
        if name.startswith("model.")
    }

    def evaluate(role, params, head):
        features = encoded[role]
        rows_data = data.splits[role]
        prediction = _predict(
            preset,
            params,
            head,
            features,
            spec.kind,
            pad_id,
            config.batch_size * 2,
        )
        if spec.kind in {"sequence", "pair"}:
            return _classification_scores(
                list(prediction), [row["label"] for row in rows_data]
            )
        if spec.kind == "token":
            tags = []
            for feature, logits in zip(features, prediction, strict=True):
                first = feature["first_token"]
                words = max(first) + 1 if first else 0
                tags.append(
                    [
                        data.labels[int(logits[first[word]])] if word in first else "O"
                        for word in range(words)
                    ]
                )
            gold = [row["tags"][: len(tags[index])] for index, row in enumerate(rows_data)]
            return {
                "entity_f1": _entity_f1(tags, gold),
                "weighted_f1": _token_weighted_f1(tags, gold),
            }
        start_logits, end_logits = prediction
        return _qa_scores(
            _qa_answers(
                features,
                start_logits,
                end_logits,
                rows_data,
                config.qa_max_answer_tokens,
            ),
            [[answer["text"] for answer in row["answers"]] for row in rows_data],
        )

    dev_by_rate = {}
    for learning_rate in config.learning_rates:
        _, _, dev = _train_fold(preset, encoder, encoded["train"], data.labels, spec.kind, config, learning_rate, config.seeds[0], lambda p, h: evaluate("dev", p, h)[spec.metric], pad_id)
        dev_by_rate[learning_rate] = dev
        log(f"{spec.name}: lr {learning_rate:g} dev {spec.metric} {dev:.2f}")
    best_lr = max(dev_by_rate, key=lambda value: dev_by_rate[value])
    result = TaskResult(spec.name, spec.metric, best_lr, dev_by_rate)
    for seed in config.seeds:
        params, head, _ = _train_fold(preset, encoder, encoded["train"], data.labels, spec.kind, config, best_lr, seed, lambda p, h: evaluate("dev", p, h)[spec.metric], pad_id)
        result.test_by_seed[seed] = evaluate("test", params, head)
        log(f"{spec.name}: seed {seed} test {result.test_by_seed[seed]}")
    result.seconds = time.monotonic() - started
    return result
