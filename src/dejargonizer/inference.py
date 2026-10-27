"""Inference API and command-line interface for a trained simplifier."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

class Dejargonizer:
    def __init__(self, checkpoint: str | Path):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        started = time.perf_counter()
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.device = str(next(self.model.parameters()).device)

    def simplify(
        self,
        text: str,
        max_input_tokens: int = 512,
        max_output_tokens: int = 256,
        return_metrics: bool = False,
    ) -> str | tuple[str, dict[str, float | int | str]]:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        import torch

        inputs = self.tokenizer(
            f"simplify: {text.strip()}",
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        input_tokens = int(inputs["input_ids"].shape[-1])
        started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=min(max_output_tokens, 256),
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
                early_stopping=True,
            )
        generation_seconds = time.perf_counter() - started
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_tokens = int(outputs.shape[-1])
        metrics = {
            "device": self.device,
            "model_load_seconds": round(self.load_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_tokens_per_second": round(
                output_tokens / generation_seconds if generation_seconds else 0.0, 2
            ),
        }
        return (result, metrics) if return_metrics else result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--checkpoint", default="artifacts/t5-small-pilot")
    args = parser.parse_args()
    model = Dejargonizer(args.checkpoint)
    result, metrics = model.simplify(args.text, return_metrics=True)
    print(result)
    print(f"Metrics: {metrics}", file=sys.stderr)


if __name__ == "__main__":
    main()
