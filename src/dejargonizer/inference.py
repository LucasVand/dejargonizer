"""Inference API and command-line interface for a trained simplifier."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path


def _process_peak_memory_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    bytes_used = usage if sys.platform == "darwin" else usage * 1024
    return round(bytes_used / (1024 * 1024), 2)


def _device_memory_metrics(torch: object) -> dict[str, float]:
    metrics: dict[str, float] = {}
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        metrics["device_memory_allocated_mb"] = round(
            cuda.memory_allocated() / (1024 * 1024), 2
        )
        metrics["device_memory_reserved_mb"] = round(
            cuda.memory_reserved() / (1024 * 1024), 2
        )
        metrics["device_memory_peak_allocated_mb"] = round(
            cuda.max_memory_allocated() / (1024 * 1024), 2
        )
        return metrics

    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "current_allocated_memory"):
        metrics["device_memory_allocated_mb"] = round(
            mps.current_allocated_memory() / (1024 * 1024), 2
        )
    return metrics


class Dejargonizer:
    def __init__(self, checkpoint: str | Path):
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        started = time.perf_counter()
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started
        self.device = str(next(self.model.parameters()).device)
        import torch

        self.process_peak_memory_after_load_mb = _process_peak_memory_mb()
        self.device_memory_after_load = _device_memory_metrics(torch)

    def simplify(
        self,
        text: str,
        max_input_tokens: int = 512,
        max_output_tokens: int = 256,
        return_metrics: bool = False,
        show_progress: bool = False,
    ) -> str | tuple[str, dict[str, float | int | str]]:
        if not text or not text.strip():
            raise ValueError("text must not be empty")
        import torch

        if show_progress:
            print("Tokenizing input...", file=sys.stderr, flush=True)
        started = time.perf_counter()
        inputs = self.tokenizer(
            f"aggressively simplify: {text.strip()}",
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        tokenization_seconds = time.perf_counter() - started
        input_tokens = int(inputs["input_ids"].shape[-1])
        if show_progress:
            print(
                f"Input tokenized in {tokenization_seconds:.2f}s "
                f"({input_tokens} tokens)",
                file=sys.stderr,
                flush=True,
            )
        if show_progress:
            print("Generating simplified text...", file=sys.stderr, flush=True)
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
        if show_progress:
            print(
                f"Generation complete in {generation_seconds:.2f}s",
                file=sys.stderr,
                flush=True,
            )
        started = time.perf_counter()
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        decoding_seconds = time.perf_counter() - started
        if show_progress:
            print(
                f"Decoded output in {decoding_seconds:.2f}s",
                file=sys.stderr,
                flush=True,
            )
        output_tokens = int(outputs.shape[-1])
        metrics = {
            "device": self.device,
            "model_load_seconds": round(self.load_seconds, 3),
            "tokenization_seconds": round(tokenization_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "decoding_seconds": round(decoding_seconds, 3),
            "total_inference_seconds": round(
                tokenization_seconds + generation_seconds + decoding_seconds, 3
            ),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_tokens_per_second": round(
                output_tokens / generation_seconds if generation_seconds else 0.0, 2
            ),
            "process_peak_memory_mb": _process_peak_memory_mb(),
            "process_peak_memory_after_load_mb": self.process_peak_memory_after_load_mb,
            "device_memory_after_load_mb": self.device_memory_after_load.get(
                "device_memory_allocated_mb", 0.0
            ),
            **_device_memory_metrics(torch),
        }
        return (result, metrics) if return_metrics else result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--checkpoint", default="artifacts/t5-small-pilot")
    args = parser.parse_args()
    print("=== DIAGNOSTICS ===", file=sys.stderr, flush=True)
    print(f"Loading model from {args.checkpoint}...", file=sys.stderr, flush=True)
    load_started = time.perf_counter()
    model = Dejargonizer(args.checkpoint)
    print(
        f"Model loaded in {time.perf_counter() - load_started:.2f}s "
        f"(device: {model.device}; "
        f"process peak memory: {model.process_peak_memory_after_load_mb:.2f} MB)",
        file=sys.stderr,
        flush=True,
    )
    inference_started = time.perf_counter()
    result, metrics = model.simplify(
        args.text, return_metrics=True, show_progress=True
    )
    print(
        f"Simplification finished; "
        f"total inference time {time.perf_counter() - inference_started:.2f}s",
        file=sys.stderr,
        flush=True,
    )
    print(
        f"Device: {metrics['device']}\n"
        f"Timing: tokenization={metrics['tokenization_seconds']:.2f}s, "
        f"generation={metrics['generation_seconds']:.2f}s, "
        f"decoding={metrics['decoding_seconds']:.2f}s, "
        f"total inference={metrics['total_inference_seconds']:.2f}s\n"
        f"Tokens: input={metrics['input_tokens']}, output={metrics['output_tokens']}, "
        f"speed={metrics['output_tokens_per_second']:.2f} tokens/s\n"
        f"Memory: process peak={metrics['process_peak_memory_mb']:.2f} MB"
        + (
            f", device allocated={metrics['device_memory_allocated_mb']:.2f} MB"
            if "device_memory_allocated_mb" in metrics
            else ""
        ),
        file=sys.stderr,
        flush=True,
    )
    print("=== END DIAGNOSTICS ===", file=sys.stderr, flush=True)
    print("=== MODEL OUTPUT ===")
    print(result)


if __name__ == "__main__":
    main()
