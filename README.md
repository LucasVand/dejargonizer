# Dejargonizer

Dejargonizer rewrites technical English in simpler English. The project uses a streamed sample
of arXiv abstracts, generates simplified targets with Ollama or Cloudflare Workers AI, fine-tunes
Google's `t5-small`, and evaluates the resulting model.

## Setup

Create or activate the project environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project:

```bash
.venv/bin/python -m pip install -e '.[dev]'
```

The scripts activate `.venv` themselves, so activating it manually is optional when using the
shell scripts.

## Configuration

The main configuration file is `configs/pilot.yaml`.

### Dataset settings

```yaml
dataset:
  name: common-pile/arxiv_abstracts
  split: train
  revision: main
  output_dir: data
  sample_size: 10000
  max_scan_records: 100000
  max_source_tokens: 200
  min_words: 50
  max_words: 500
  seed: 42
  shuffle_buffer: 1000
```

- `name`, `split`, and `revision` identify the Hugging Face dataset.
- `sample_size` is the maximum number of source records written locally.
- `max_scan_records` is the maximum number of remote records examined. It must be at least
  `sample_size`.
- `offset` skips that many records in the deterministic shuffled stream before scanning. The
  scan window is `[offset, offset + max_scan_records)`, which lets separate datasets use
  non-overlapping windows.
- `min_words` and `max_words` filter source abstracts.
- `max_source_tokens` is an additional source-length safety limit.
- `seed` and `shuffle_buffer` control deterministic streamed selection.

The data loader always uses streaming mode. It never downloads or caches the complete arXiv
dataset. Only the selected records are written to `data/interim/pilot.jsonl`.

To create two disjoint source datasets with the same configuration, use separate config copies:

```yaml
# configs/pilot-a.yaml
dataset:
  offset: 0
  sample_size: 10000
  max_scan_records: 100000
```

```yaml
# configs/pilot-b.yaml
dataset:
  offset: 100000
  sample_size: 10000
  max_scan_records: 100000
```

Run each config separately and write each result to a different output directory. Keep the same
dataset revision, seed, shuffle buffer, filters, and quota settings. The second window begins
after the first window's entire scan range, so selected records cannot overlap between the two
datasets.

### Sentence-length quotas

The current configuration assigns one slice to each selected abstract:

```yaml
chunks:
  enabled: true
  max_sentences: 8
  quota_percentages: [11.11, 11.11, 11.11, 11.11, 11.11, 11.11, 11.11, 11.11, 11.12]
```

The nine quota categories are, in order:

1. one sentence
2. two sentences
3. three sentences
4. four sentences
5. five sentences
6. six sentences
7. seven sentences
8. eight sentences
9. the full abstract

Each source abstract is used once. For a sentence category, one deterministic contiguous section
is selected. The full abstract is used for the ninth category. The percentages must total 100%.
If an abstract is too short for a requested category, the sampler uses another category that the
abstract can satisfy.

### Labeling settings

```yaml
labeling:
  provider: ollama
  endpoint: http://100.97.198.56:11434
  model: llama3.1:8b
  prompt_version: v1
  temperature: 0.2
  max_tokens: 300
  timeout: 300
```

- `provider` is either `ollama` or `cloudflare`.
- `endpoint` is used by Ollama and should point to an Ollama server reachable from this machine.
- `model` must be installed on the selected provider.
- `temperature` controls generation variability.
- `max_tokens` limits the generated simplification length.
- `timeout` is the HTTP timeout in seconds for Ollama.

For Cloudflare, set `provider: cloudflare` and use a Cloudflare model name. The ignored local
`start.sh` file must export `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`.

### Training settings

```yaml
model:
  checkpoint: t5-small
  max_input_tokens: 512
  max_target_tokens: 256

training:
  output_dir: artifacts/t5-small-pilot
  learning_rate: 0.0003
  epochs: 3
  batch_size: 1
  gradient_accumulation_steps: 16
  seed: 42
```

The model and tokenizer are saved to `training.output_dir`. Intermediate Trainer checkpoints
are saved under that directory as `checkpoint-*`.

## Scripts

### `generate_data.sh`

Creates the bounded source dataset:

```bash
./generate_data.sh
```

It activates `.venv`, reads `configs/pilot.yaml`, streams arXiv, reports scanning progress, and
writes:

```text
data/interim/pilot.jsonl
data/interim/manifest.json
```

### `generate_pairs.sh`

Generates simplified source/target pairs. It is resumable: existing records are identified by
source hash and skipped.

Using the configured provider:

```bash
./generate_pairs.sh
```

Only validation records:

```bash
./generate_pairs.sh --validation
```

Only test records:

```bash
./generate_pairs.sh --test
```

Force Ollama regardless of the configured provider:

```bash
./generate_pairs.sh --ollama
./generate_pairs.sh --validation --ollama
./generate_pairs.sh --test --ollama
```

Generated labels are appended to:

```text
data/labeled/pilot.jsonl
```

Every record stores the source, target, split, provider, model, prompt version, source hash, and
validation issues. A failed request is recorded with `issues: ["labeling_error"]`; rerunning the
script currently treats that recorded hash as already handled, so remove the failed record or
create a cleaned output before retrying it.

### `check_similarity.sh`

Checks the labeled source/target pairs with Sentence Transformers:

```bash
./check_similarity.sh
```

It compares the original source with the generated target, not the fine-tuned T5 output. It
prints every pair below the threshold and writes:

```text
data/labeled/similarity.json
```

### `verify_pairs.sh`

Checks that the generated JSONL pair file is structurally valid:

```bash
./verify_pairs.sh
```

The checker validates JSON syntax, required fields, field types, valid split names, non-empty
source and target text, source hashes, duplicate IDs, and malformed records. It exits with a
nonzero status and reports line numbers when the file is broken.

### `combine_pairs.sh`

Combines generated pair files into one output file without modifying either input:

```bash
./combine_pairs.sh data/labeled/pilot-a.jsonl data/labeled/pilot-b.jsonl \
  --output data/labeled/pilot-combined.jsonl
```

Inputs are checked before combining. When records share an ID or source hash, the first occurrence
wins: duplicates later in the same file or in later input files are skipped, even when their
targets differ. Malformed records and other invalid pair-file errors stop the merge without
replacing the output.

### `train_model.sh`

Fine-tunes T5-small:

```bash
./train_model.sh
```

It uses accepted records (`issues: []`) from the train and validation splits. The final model is
saved to:

```text
artifacts/t5-small-pilot/
```

Training requires accepted train and validation pairs. Test pairs are not required to start
training, but are needed for an unbiased final evaluation.

### `test.sh`

Evaluates the saved model on accepted test pairs:

```bash
./test.sh
```

It runs SARI and readability evaluation through `dejargonizer-evaluate`. Generate test pairs first
with `./generate_pairs.sh --test`.

### `simplify.sh`

Runs the saved model on one input:

```bash
./simplify.sh "A core challenge of multi-robot interactions is collision avoidance"
```

The checkpoint is loaded from `artifacts/t5-small-pilot`.

## Restarting and rebuilding data

### Resume an interrupted pair-generation run

Run the same command again:

```bash
./generate_pairs.sh
```

It skips source hashes already present and continues with pending records. To resume only a
specific split:

```bash
./generate_pairs.sh --validation
./generate_pairs.sh --test
```

With Ollama:

```bash
./generate_pairs.sh --ollama
./generate_pairs.sh --validation --ollama
./generate_pairs.sh --test --ollama
```

### Relabel existing source records

To keep the same arXiv source records but generate new targets with a new model or prompt, remove
only the labeled outputs:

```bash
rm -f data/labeled/pilot.jsonl
rm -f data/labeled/similarity.json
./generate_pairs.sh
```

### Create a completely new source dataset

To apply changed dataset, filtering, quota, or sampling configuration:

```bash
rm -f data/interim/pilot.jsonl
rm -f data/interim/manifest.json
rm -f data/labeled/pilot.jsonl
rm -f data/labeled/similarity.json
rm -rf artifacts/t5-small-pilot
```

Then recreate the source records and labels:

```bash
./generate_data.sh
./generate_pairs.sh
./generate_pairs.sh --validation
./generate_pairs.sh --test
```

Only remove `artifacts/t5-small-pilot` if you also want to discard the old trained model. The
`data/` and `artifacts/` directories are ignored by Git.

## Typical end-to-end workflow

```bash
# 1. Configure configs/pilot.yaml.
./generate_data.sh

# 2. Generate all train pairs, then the held-out splits.
./generate_pairs.sh
./generate_pairs.sh --validation
./generate_pairs.sh --test

# 3. Review semantic similarity of labels.
./check_similarity.sh

# 4. Fine-tune T5-small.
./train_model.sh

# 5. Evaluate and run inference.
./test.sh
./simplify.sh "Technical text to simplify"
```

## Important files

```text
configs/pilot.yaml                 Main configuration
data/interim/pilot.jsonl           Streamed source records
data/labeled/pilot.jsonl           Source/target training pairs
data/labeled/similarity.json       Label similarity report
artifacts/t5-small-pilot/          Trained model and checkpoints
start.sh                           Local Cloudflare credentials; ignored by Git
```

Never commit `start.sh`, API tokens, generated datasets, model artifacts, or `.venv`.
