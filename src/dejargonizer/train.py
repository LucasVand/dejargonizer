"""Fine-tune T5-small on validated simplification pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_examples(path: str | Path) -> dict[str, Dataset]:
    from datasets import Dataset

    groups: dict[str, list[dict[str, str]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    total_by_split = {split: 0 for split in groups}
    accepted_by_split = {split: 0 for split in groups}
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            split = item.get("split")
            if split not in groups:
                continue
            total_by_split[split] += 1
            if not item.get("issues"):
                accepted_by_split[split] += 1
                groups[split].append(
                    {"source": item["source"], "target": item["target"]}
                )
    if not groups["train"] or not groups["validation"]:
        raise ValueError(
            "Need accepted train and validation examples; "
            f"found total={total_by_split}, accepted={accepted_by_split}. "
            "Resume labeling until validation records are present."
        )
    return {split: Dataset.from_list(rows) for split, rows in groups.items() if rows}


def train(input_path: str | Path, config: dict[str, Any]) -> None:
    from transformers import (
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        T5ForConditionalGeneration,
        T5Tokenizer,
        set_seed,
    )

    model_config, train_config = config["model"], config["training"]
    set_seed(int(train_config["seed"]))
    tokenizer = T5Tokenizer.from_pretrained(model_config["checkpoint"])
    model = T5ForConditionalGeneration.from_pretrained(model_config["checkpoint"])
    datasets = load_examples(input_path)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        inputs = tokenizer(
            [f"simplify: {text}" for text in batch["source"]],
            max_length=model_config["max_input_tokens"],
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=model_config["max_target_tokens"],
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = {
        split: dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        for split, dataset in datasets.items()
    }
    args = Seq2SeqTrainingArguments(
        output_dir=train_config["output_dir"],
        learning_rate=float(train_config["learning_rate"]),
        num_train_epochs=float(train_config["epochs"]),
        per_device_train_batch_size=int(train_config["batch_size"]),
        per_device_eval_batch_size=int(train_config["batch_size"]),
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        report_to="none",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    trainer.train()
    trainer.save_model(train_config["output_dir"])
    tokenizer.save_pretrained(train_config["output_dir"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/labeled/pilot.jsonl")
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    import yaml

    with Path(args.config).open() as file:
        train(args.input, yaml.safe_load(file))


if __name__ == "__main__":
    main()
