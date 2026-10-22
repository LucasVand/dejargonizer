"""Inference API and command-line interface for a trained simplifier."""

from __future__ import annotations

import argparse
from pathlib import Path

class Dejargonizer:
    def __init__(self, checkpoint: str | Path):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)

    def simplify(self, text: str, max_input_tokens: int = 512, max_output_tokens: int = 256) -> str:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        inputs = self.tokenizer(
            f"simplify: {text.strip()}",
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        outputs = self.model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=min(max_output_tokens, 256),
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--checkpoint", default="artifacts/t5-small-pilot")
    args = parser.parse_args()
    print(Dejargonizer(args.checkpoint).simplify(args.text))


if __name__ == "__main__":
    main()
