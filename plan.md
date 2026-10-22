# Dejargonizer: Technical English to Simple English

## 1. Project goal

Build a text-to-text translation model that takes technical English, especially scientific writing, and rewrites it in clear, simple English for a general audience. The rewrite should:

- Preserve the original meaning, important facts, qualifications, and uncertainty.
- Prefer familiar words and shorter, clearer sentences.
- Explain or replace jargon where possible.
- Avoid inventing facts, dropping important constraints, or making claims stronger than the source.

The first model will be based on Google’s **T5-small** and fine-tuned for supervised simplification. Because the source dataset does not contain simplified references, Cloudflare Workers AI will generate the training targets from arXiv abstracts.

## 2. Initial scope and assumptions

The first release should be a small pilot rather than an attempt to process the entire arXiv dataset. This keeps API cost, annotation time, and iteration speed manageable.

Recommended pilot:

- Sample 5,000–20,000 abstracts after filtering.
- Keep abstracts in English and within a practical length range, such as 50–500 words.
- Deduplicate near-identical abstracts before creating labels.
- Split the source abstracts into train, validation, and test sets before calling the labeling API.
- Use the same source text only once across all splits.
- Generate labels for the training and validation sets first; hold out a manually reviewed test set where possible.

The generated simplifications are **silver labels**, not authoritative human references. The plan therefore includes quality checks and human evaluation before treating the model as reliable.

## 3. Proposed repository structure

Create the following structure as the implementation is built:

```text
dejargonizer/
├── data/
│   ├── raw/              # Only the bounded pilot records; never the full source dataset
│   ├── interim/          # Filtered, deduplicated, split records
│   ├── labeled/          # Cloudflare Workers AI-generated simplification pairs
│   └── processed/        # Tokenizer-ready datasets
├── configs/
│   └── pilot.yaml        # Dataset, model, and training settings
├── src/
│   └── dejargonizer/
│       ├── data.py       # Streaming, bounded sampling, filtering, splitting, and formatting
│       ├── labeling.py   # Cloudflare Workers AI labeling client and retry handling
│       ├── quality.py    # Validation and filtering of generated targets
│       ├── train.py      # T5 fine-tuning entry point
│       ├── evaluate.py   # Automated and readability evaluation
│       └── inference.py  # Single-text and batch inference
├── tests/
├── notebooks/            # Optional exploratory analysis only
├── requirements.txt      # Or pyproject.toml
└── plan.md
```

Do not commit API keys, raw credentials, or downloaded dataset artifacts. Store secrets in
environment variables `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`. The data pipeline must never download
or cache the entire arXiv dataset.

## 4. Data acquisition and preparation

### 4.0 Dataset access constraint

The full dataset must **never be downloaded**. Use Hugging Face streaming mode (for example,
`load_dataset(..., streaming=True)`) or an equivalent range/streaming mechanism so records are
read incrementally from the remote dataset. Configure the client to avoid materializing the
full dataset in the local Hugging Face cache.

The pipeline must:

- Select only the bounded number of records required for the pilot.
- Stop reading as soon as the configured sample has been collected, rather than iterating to the
  end of the dataset.
- Write only the selected records and their required metadata to `data/raw/` or `data/interim/`.
- Use a deterministic selection strategy where the remote dataset supports reliable shuffling or
  where a documented bounded scan and seed-based selection is sufficient.
- Refuse configurations that request the full dataset or omit a maximum record/token limit.
- Report the number of records and approximate source tokens read, retained, and discarded.
- Keep temporary files bounded and remove partial artifacts after failed runs.

Because exact random access may not be available in streaming mode, document the tradeoff between
reproducibility and remote scan size. Pin the dataset revision and seed, and save the selected
record IDs and hashes so later stages can resume without rereading or relabeling them. If a
streaming shuffle buffer is used, configure its maximum size explicitly; it must remain much
smaller than the full dataset.

### 4.1 Read the arXiv abstracts as a stream

Use the Hugging Face `datasets` library in streaming mode to load the
`common-pile/arxiv_abstracts` dataset. Confirm the dataset configuration, split names, and field
names programmatically rather than hard-coding assumptions from the web viewer. Do not call a
non-streaming load that downloads an Arrow/Parquet copy of the complete split.

Record:

- Dataset revision or download date.
- Dataset license and usage requirements.
- Source field used as the input abstract.
- Any paper identifier or metadata retained for traceability.
- Streaming configuration, maximum scan/record/token limits, and selection seed.

### 4.2 Filter and normalize inputs

Apply conservative preprocessing:

- Keep English-language records.
- Remove missing, empty, or obviously malformed abstracts.
- Normalize whitespace without destroying punctuation or mathematical notation.
- Reject extremely short and excessively long records.
- Preserve abbreviations, numbers, units, and symbols that carry meaning.
- Detect and remove exact duplicates and near duplicates.
- Flag abstracts containing dense equations, markup, or unusual formatting for separate analysis.

Do not simplify or rewrite the source before sending it to the labeling model. The original source must remain available for auditing.

### 4.3 Split before synthetic labeling

Create deterministic train/validation/test splits using a fixed seed. Prefer grouping by paper or metadata when related records could otherwise cross splits. Save the split assignment with every record.

Suggested initial proportions:

- 80% training
- 10% validation
- 10% test

For the test split, generate targets only after the pipeline is stable, and retain a manually reviewed subset of at least 200 examples. This prevents tuning entirely against synthetic targets.

## 5. Generate simplified targets with Cloudflare Workers AI

### 5.1 Label schema

Store one JSON record per example with at least:

```json
{
  "id": "stable-source-id",
  "split": "train",
  "source": "Original technical abstract",
  "target": "Plain-English rewrite",
  "label_model": "configured-model-name",
  "prompt_version": "v1",
  "status": "accepted"
}
```

Keep the model name, prompt version, timestamp, request ID if available, and any validation error. This makes labels reproducible and allows later relabeling without mixing incompatible target versions.

### 5.2 Prompt requirements

Use a system instruction that tells the model to:

1. Rewrite for an educated general audience with no specialist background.
2. Use common words and direct sentence structure.
3. Explain or replace specialized terms when doing so does not reduce accuracy.
4. Preserve entities, numbers, dates, units, relationships, uncertainty, and causal limitations.
5. Avoid adding information that is not present in the abstract.
6. Keep the result concise but complete.
7. Return only the simplified text in a strict structured response.

Use a low-temperature configuration for consistency. Version the prompt and model settings. Do not ask the labeling model to summarize: the task is meaning-preserving simplification, and summaries may omit critical details.

### 5.3 Cost, reliability, and privacy controls

Implement:

- Batched or rate-limited requests within provider limits.
- Exponential backoff for transient failures.
- A bounded retry count and explicit failed-record output.
- Caching keyed by source hash, prompt version, and model version.
- A dry-run mode that estimates token usage and cost.
- Resume support so an interrupted run does not relabel completed records.
- Redaction or review of any metadata that should not be sent to an external API.

The source abstracts are public research text, but the implementation should still document that an external API is being used and should avoid sending unnecessary metadata.

## 6. Target quality control

Synthetic labels should be validated before training. Add automatic checks to reject or flag targets when:

- The target is empty, malformed, or nearly identical to the source when substantial simplification was expected.
- The target is dramatically shorter and appears to omit key content.
- Numbers, units, named entities, negations, or uncertainty markers are missing or changed unexpectedly.
- The target contains meta-commentary such as “Here is a simplified version”.
- The output includes unsupported claims or obvious hallucinations.
- The target is not in English or violates the output format.
- The target is excessively long or contains repeated text.

Use a combination of string-level checks and a semantic consistency check. Do not automatically trust an LLM judge as the only correctness gate. Sample accepted, rejected, and borderline records for human review and revise the prompt or filters based on observed failure modes.

## 7. Fine-tune T5-small

Use the Hugging Face Transformers and Datasets libraries with the `t5-small` checkpoint.

Before fine-tuning, run the labeled-data semantic check:

```bash
dejargonizer-similarity \
  --input data/labeled/pilot.jsonl \
  --report data/labeled/similarity.json \
  --threshold 0.70
```

This uses `sentence-transformers/all-MiniLM-L6-v2` to compare each original abstract with
its generated simplification. It evaluates the source/target labels directly, not output from
the tuned T5 model. Review pairs below the threshold before including them in training; the
threshold is a review flag, not proof that a pair is correct or incorrect.

### 7.1 Input/output format

Prefix each input with a task label, for example:

```text
simplify: <technical abstract>
```

The target is the generated plain-English text. Tokenize source and target separately with the T5 tokenizer, apply explicit maximum lengths, and track truncation rates. If many abstracts are truncated, adjust filtering or use a chunking strategy rather than silently losing content.

### 7.2 Training baseline

Start with a reproducible baseline:

- Fixed random seed.
- AdamW optimizer.
- Small learning-rate sweep rather than one unverified value.
- Gradient accumulation if GPU memory is limited.
- Mixed precision when supported.
- Early stopping based on validation loss and simplification metrics.
- Best-checkpoint saving.
- Logging of configuration, dataset revision, label model, prompt version, and code revision.

Use the validation set for model and hyperparameter selection. Do not repeatedly tune on the held-out human-reviewed test set.

### 7.3 Baselines

Compare the fine-tuned model against:

- The unchanged technical input.
- A simple rule-based baseline, such as sentence splitting and a controlled vocabulary transform if practical.
- The OpenAI labeling model used directly, on a small evaluation sample.

The T5 model should be judged not only by similarity to synthetic targets but also by clarity and preservation of meaning.

## 8. Evaluation

### 8.1 Automated metrics

Report metrics on a fixed test set:

- **SARI** for measuring appropriate additions, deletions, and keeps in simplification.
- **BLEU** and/or **ROUGE** as secondary overlap metrics, with the limitation that they do not reliably measure simplicity or factual correctness.
- Source-target length and compression statistics.
- Entity, number, unit, and negation preservation rates.

If only one synthetic reference exists, explicitly state that metric scores are limited by reference quality. Multiple references for a smaller test subset would improve evaluation.

### 8.2 Readability

Calculate readability before and after simplification, such as:

- Flesch Reading Ease.
- Flesch–Kincaid grade level.
- Average sentence length.
- Average word length or syllable count.

Set target ranges only after inspecting the source distribution. A lower grade level is not sufficient if meaning is lost.

### 8.3 Human evaluation

Create an evaluation rubric and have reviewers compare source and output. At minimum, score:

1. Meaning preservation.
2. Factual accuracy.
3. Clarity for a non-specialist.
4. Reduction of unnecessary jargon.
5. Grammatical quality.

Use blinded examples that mix the T5 output, baseline output, and reference labels where appropriate. Have at least two reviewers score a subset and measure agreement. Record and categorize severe errors, especially hallucinations, reversed meaning, missing limitations, and incorrect numbers.

Define a release gate before deployment, for example requiring high meaning-preservation and factual-accuracy scores, not merely improved readability.

## 9. Inference interface

Implement a small inference layer that:

- Loads the fine-tuned tokenizer and model from a configured checkpoint.
- Accepts one text or a batch of texts.
- Applies the exact same `simplify:` prefix and tokenization settings used in training.
- Supports configurable generation parameters such as beam count, maximum output length, and length penalty.
- Returns the simplified text plus optional metadata such as model version.
- Rejects empty input and clearly reports overlong input.
- Runs in CPU mode for local use and supports GPU when available.

Begin with a Python API and command-line interface. Add an HTTP service only after the model behavior is stable. The interface should never imply that the output is guaranteed correct; document that users should review high-stakes rewrites.

## 10. Testing and reproducibility

Add tests for:

- Dataset field discovery and filtering.
- Deterministic splitting.
- Prompt construction and structured response parsing.
- Retry, caching, and resume behavior in the labeling pipeline.
- Quality-control checks for changed numbers, entities, and negations.
- Tokenization and truncation behavior.
- Inference on a known fixture.
- Loading a saved checkpoint and producing non-empty output.

Maintain a small versioned fixture dataset for fast tests. Keep full training and API-labeling runs separate from the normal test suite.

## 11. Implementation milestones

1. **Project setup:** Create the Python package, configuration system, environment-variable handling, and tests.
2. **Dataset inspection:** Stream only a bounded sample from the Hugging Face dataset, confirm schema/license, and implement filtering without downloading the full split.
3. **Split and cache:** Produce deterministic pilot splits and a manifest with stable IDs and hashes; cache only those selected records.
4. **Labeling pipeline:** Add the Cloudflare Workers AI client, versioned prompt, structured output parsing, retries, caching, and cost estimates.
5. **Label review:** Run quality checks and manually inspect a representative sample; revise the prompt before generating the full pilot labels.
6. **Training baseline:** Fine-tune T5-small and save reproducible checkpoints and logs.
7. **Evaluation:** Run automated metrics, readability analysis, and human evaluation on the held-out set.
8. **Inference:** Package a Python API and CLI using the best validated checkpoint.
9. **Iteration:** Analyze errors, improve data or prompts, retrain, and compare against the prior checkpoint.

## 12. Risks and decisions to revisit

- **Synthetic-target bias:** T5 may learn the labeling model’s style and mistakes rather than robust simplification. Add human-written references for a smaller benchmark as soon as possible.
- **Meaning loss:** Aggressive simplification can remove caveats or technical precision. Treat factual and meaning-preservation scores as release blockers.
- **Domain mismatch:** arXiv abstracts cover many fields and may not represent the intended users. Track subject areas and evaluate by domain.
- **Long inputs:** T5-small has practical context limits. Measure truncation and define a policy for long documents before exposing the model to users.
- **Metric mismatch:** Better overlap or readability scores can coexist with worse writing. Use human evaluation and error analysis.
- **API dependency and cost:** Label generation depends on an external provider. Cache every result, make the provider configurable, and preserve failed records for later retry.
- **Dataset and model licensing:** Verify the dataset terms, T5 model terms, and Cloudflare Workers AI usage terms before redistributing generated labels or model artifacts.

## 13. Definition of done for the pilot

The pilot is complete when:

- A documented subset of arXiv abstracts has deterministic train/validation/test membership.
- The pipeline demonstrably uses streaming/bounded access and never downloads or caches the entire dataset.
- Every accepted training example has traceable source, target, model, prompt, and validation metadata.
- T5-small fine-tunes successfully from a clean environment using the documented configuration.
- Evaluation includes SARI or an equivalent simplification metric, readability changes, and human ratings for meaning and clarity.
- The trained checkpoint can be loaded through a Python API and CLI.
- Known failure modes and limitations are documented, and the model is not presented as safe for unsupervised high-stakes rewriting.
