# Europeana Post-Correct Data

This repository builds **ChronoCorrect-Europeana-FR**, a silver dataset for historical OCR post-correction derived from [`biglam/europeana_newspapers`](https://huggingface.co/datasets/biglam/europeana_newspapers).

The goal is to create OCR correction examples that are useful not only for character-level correction, but also for **historical NLP evaluation**: named entity preservation, date and number preservation, hallucination risk, and overcorrection detection.

## Installation

Create and activate an environment, then install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```

## API keys

The script reads API keys from environment variables.

For OpenAI:

```bash
export OPENAI_API_KEY="sk-..."
```

For Hugging Face upload:

```bash
export HF_TOKEN="hf_..."
```

Do **not** commit API keys, `.env` files, generated JSONL files, or local dataset folders.

## Quick dry run without API

Use this to test loading, filtering, splitting, and writing without spending OpenAI credits:

```bash
python build_dataset.py \
  --output-jsonl outputs/test_no_api.jsonl \
  --max-examples 20 \
  --max-pages 500 \
  --language fr \
  --no-api \
  --verbose
```

This writes placeholder examples where `corrected_text` equals `ocr_text`.

## Generate examples with OpenAI

Small test run:

```bash
export OPENAI_API_KEY="sk-..."

python build_dataset.py \
  --output-jsonl outputs/chronocorrect_europeana_fr_test.jsonl \
  --max-examples 20 \
  --max-pages 500 \
  --language fr \
  --model-correction gpt-5-mini \
  --model-annotation gpt-5-mini \
  --resume \
  --verbose
```

Larger run:

```bash
python build_dataset.py \
  --output-jsonl outputs/chronocorrect_europeana_fr.jsonl \
  --max-examples 1000 \
  --max-pages 50000 \
  --language fr \
  --model-correction gpt-5-mini \
  --model-annotation gpt-5-mini \
  --resume \
  --verbose
```

## Continue from where the script stopped

Use `--resume` and `--target-total-examples`.

For example, if the JSONL already contains 20 examples and you want to continue until it contains 200 total examples:

```bash
python build_dataset.py \
  --output-jsonl outputs/chronocorrect_europeana_fr_test.jsonl \
  --target-total-examples 200 \
  --max-pages 10000 \
  --language fr \
  --model-correction gpt-5-mini \
  --model-annotation gpt-5-mini \
  --resume \
  --verbose
```

`--target-total-examples` means total valid JSONL records, not newly generated records.

## Export to Hugging Face format

After generation, export the JSONL to a local Hugging Face `DatasetDict`:

```bash
python build_dataset.py \
  --output-jsonl outputs/chronocorrect_europeana_fr_test.jsonl \
  --export-only \
  --export-hf \
  --hf-output-dir hf_dataset
```

This creates train/validation/test splits using a flat schema.

## Push to Hugging Face Hub

Push the dataset to your Hugging Face account:

```bash
export HF_TOKEN="hf_..."

python build_dataset.py \
  --output-jsonl outputs/chronocorrect_europeana_fr_test.jsonl \
  --export-only \
  --export-hf \
  --push-to-hub \
  --hub-dataset-id EmanuelaBoros/chronocorrect-europeana-fr \
  --private
```

Remove `--private` when the dataset is ready to be public.

## Dataset fields

The exported HF dataset contains the following main fields:

| Field | Description |
|---|---|
| `id` | Stable example identifier |
| `source_dataset` | Source dataset name |
| `source_id` | Source document/page identifier |
| `paragraph_id` | Paragraph-level identifier |
| `language` | Language code, currently usually `fr` |
| `title` | Newspaper title if available |
| `date` | Publication date if available |
| `mean_ocr` | Mean OCR confidence if available |
| `std_ocr` | OCR confidence standard deviation if available |
| `ocr_text` | Original OCR paragraph |
| `corrected_text` | Conservative post-OCR correction candidate |
| `correction_policy` | Currently `conservative` |
| `annotation_status` | `silver` or `silver_needs_review` |
| `selection_reasons_json` | JSON string explaining why the paragraph was selected |
| `pre_correction_features_json` | JSON string with entities/dates/numbers before correction |
| `post_correction_features_json` | JSON string with entities/dates/numbers after correction |
| `llm_annotation_json` | JSON string with correction tags and semantic change annotation |
| `automatic_risk_flags_json` | JSON string with hallucination/overcorrection/risk flags |
| `human_verification_json` | JSON string with manual verification placeholders |
| `source_metadata_json` | JSON string with source metadata |

To read JSON metadata after loading from Hugging Face:

```python
import json
from datasets import load_dataset

ds = load_dataset("EmanuelaBoros/chronocorrect-europeana-fr")
ex = ds["train"][0]

llm_annotation = json.loads(ex["llm_annotation_json"])
risk_flags = json.loads(ex["automatic_risk_flags_json"])
```

## Correction policy

The correction target is **conservative**:

- correct OCR errors only;
- preserve historical spelling when plausible;
- preserve wording, abbreviations, punctuation style, and typography when plausible;
- do not modernize;
- do not paraphrase;
- do not add missing information;
- do not remove historically meaningful content.

## Risks

Examples can be automatically marked as needing review when the script detects:

- entity count changes;
- date count changes;
- number count changes;
- unusually large length changes;
- possible hallucination according to the LLM annotation;
- possible overcorrection according to the LLM annotation;
- uncertain readings.

These examples receive:

```text
annotation_status = silver_needs_review
```

## Human verification

The script creates placeholders for later manual validation:

```json
{
  "faithful": null,
  "historical_language_preserved": null,
  "entities_correct": null,
  "dates_numbers_correct": null,
  "hallucination_present": null,
  "overcorrection_present": null,
  "notes": null
}
```

A future annotation step can fill these fields and create a small gold subset.

## Limitations

This is a silver dataset.
