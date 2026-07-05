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



