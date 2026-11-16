"""Interactive prompt session for a trained simplifier."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dejargonizer.inference import Dejargonizer

BAR_WIDTH = 32
BAR_FULL = "█"
BAR_EMPTY = "░"
ANIMATE_INTERVAL_SECONDS = 0.05
ANIMATE_CAP = 0.95


def _render_progress_bar(fraction: float, elapsed_seconds: float) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * BAR_WIDTH))
    bar = BAR_FULL * filled + BAR_EMPTY * (BAR_WIDTH - filled)
    return f"\r[{bar}] {fraction * 100:5.1f}% ({elapsed_seconds:4.1f}s)"


class LoadingBar:
    def __init__(self, stream: object | None = None):
        self._stream = stream if stream is not None else sys.stderr
        self._fraction = 0.0
        self._started = time.perf_counter()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._animate, daemon=True)

    def _animate(self) -> None:
        while not self._done.is_set():
            with self._lock:
                target = ANIMATE_CAP if self._fraction < ANIMATE_CAP else self._fraction
                self._fraction = min(
                    target,
                    self._fraction
                    + (target - self._fraction) * 0.5
                    + ANIMATE_INTERVAL_SECONDS * 0.15,
                )
                self._draw(self._fraction)
            self._done.wait(ANIMATE_INTERVAL_SECONDS)

    def _draw(self, fraction: float) -> None:
        elapsed = time.perf_counter() - self._started
        print(_render_progress_bar(fraction, elapsed), end="", flush=True, file=self._stream)

    def bump(self, fraction: float) -> None:
        with self._lock:
            self._fraction = max(self._fraction, fraction)

    def __enter__(self) -> "LoadingBar":
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._done.set()
        self._thread.join()
        self._draw(1.0)
        print(file=self._stream, flush=True)


def _load_model(checkpoint: str) -> Dejargonizer:
    from dejargonizer.inference import Dejargonizer

    with LoadingBar() as bar:
        model = Dejargonizer(checkpoint)
        bar.bump(ANIMATE_CAP)
    return model


def _run_session(model: Dejargonizer) -> None:
    print(
        f"Dejargonizer ready (device: {model.device}, "
        f"loaded in {model.load_seconds:.2f}s)."
    )
    print("Enter text to simplify. Commands: /timing, quit")
    timing = False
    while True:
        try:
            text = input("\nsimplify> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit", "q"}:
            break
        if text == "/timing":
            timing = not timing
            print(f"timing {'on' if timing else 'off'}")
            continue
        try:
            result, metrics = model.simplify(text, return_metrics=True)
        except ValueError as exc:
            print(f"error: {exc}")
            continue
        print(result)
        if timing:
            print(
                f"[{metrics['generation_seconds']:.2f}s, "
                f"{metrics['output_tokens']} out tokens, "
                f"{metrics['output_tokens_per_second']:.2f} tok/s]",
                file=sys.stderr,
                flush=True,
            )
    print("Bye!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive dejargonizer prompt.")
    parser.add_argument("--checkpoint", default="artifacts/t5-small-pilot")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"error: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        raise SystemExit(1)

    try:
        model = _load_model(args.checkpoint)
    except Exception as exc:
        print(f"\nerror: failed to load model: {exc}", file=sys.stderr)
        raise SystemExit(1)

    _run_session(model)


if __name__ == "__main__":
    main()
