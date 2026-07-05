# Europeana Post-Correct Data

This repository builds **ChronoCorrect-Europeana-FR**, a silver dataset for historical OCR post-correction derived from [`biglam/europeana_newspapers`](https://huggingface.co/datasets/biglam/europeana_newspapers).

The goal is to create OCR correction examples that are useful not only for character-level correction, but also for **historical NLP evaluation**: named entity preservation, date and number preservation, hallucination risk, and overcorrection detection.

## Installation

Create and activate an environment, then install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_md
```
