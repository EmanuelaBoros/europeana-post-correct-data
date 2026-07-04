#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import login


def as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def as_float_or_none(x: Any):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


def to_json_string(x: Any) -> str:
    if x is None:
        x = {}
    return json.dumps(x, ensure_ascii=False)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flatten complex nested fields into JSON strings.
    This avoids HF feature mismatches across train/validation/test.
    """

    return {
        "id": as_str(record.get("id")),
        "source_dataset": as_str(record.get("source_dataset")),
        "source_id": as_str(record.get("source_id")),
        "paragraph_id": as_str(record.get("paragraph_id")),
        "language": as_str(record.get("language")),
        "title": as_str(record.get("title")),
        "date": as_str(record.get("date")),
        "mean_ocr": as_float_or_none(record.get("mean_ocr")),
        "std_ocr": as_float_or_none(record.get("std_ocr")),
        "ocr_text": as_str(record.get("ocr_text")),
        "corrected_text": as_str(record.get("corrected_text")),
        "correction_policy": as_str(record.get("correction_policy")),
        "annotation_status": as_str(record.get("annotation_status")),
        # Complex fields stored safely as JSON strings
        "selection_reasons_json": to_json_string(record.get("selection_reasons")),
        "pre_correction_features_json": to_json_string(
            record.get("pre_correction_features")
        ),
        "post_correction_features_json": to_json_string(
            record.get("post_correction_features")
        ),
        "llm_annotation_json": to_json_string(record.get("llm_annotation")),
        "automatic_risk_flags_json": to_json_string(record.get("automatic_risk_flags")),
        "human_verification_json": to_json_string(record.get("human_verification")),
        "source_metadata_json": to_json_string(record.get("source_metadata")),
    }


def chronocorrect_flat_features() -> Features:
    return Features(
        {
            "id": Value("string"),
            "source_dataset": Value("string"),
            "source_id": Value("string"),
            "paragraph_id": Value("string"),
            "language": Value("string"),
            "title": Value("string"),
            "date": Value("string"),
            "mean_ocr": Value("float64"),
            "std_ocr": Value("float64"),
            "ocr_text": Value("string"),
            "corrected_text": Value("string"),
            "correction_policy": Value("string"),
            "annotation_status": Value("string"),
            "selection_reasons_json": Value("string"),
            "pre_correction_features_json": Value("string"),
            "post_correction_features_json": Value("string"),
            "llm_annotation_json": Value("string"),
            "automatic_risk_flags_json": Value("string"),
            "human_verification_json": Value("string"),
            "source_metadata_json": Value("string"),
        }
    )


def build_dataset_dict(
    jsonl_path: str,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    print(f"[load] Reading JSONL: {jsonl_path}")
    records = load_jsonl(jsonl_path)
    print(f"[load] Loaded records: {len(records)}")

    records = [normalize_record(r) for r in records]

    random.Random(seed).shuffle(records)

    n_total = len(records)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * validation_ratio)

    # For very small datasets, make sure splits are not weird.
    # If fewer than 10 records, keep everything in train.
    if n_total < 10:
        n_test = 0
        n_val = 0

    test_records = records[:n_test]
    val_records = records[n_test : n_test + n_val]
    train_records = records[n_test + n_val :]

    features = chronocorrect_flat_features()

    dataset_dict = DatasetDict()
    dataset_dict["train"] = Dataset.from_list(train_records, features=features)

    if val_records:
        dataset_dict["validation"] = Dataset.from_list(val_records, features=features)

    if test_records:
        dataset_dict["test"] = Dataset.from_list(test_records, features=features)

    return dataset_dict


def write_readme(out_dir: str):
    readme = """---
language:
- fr
task_categories:
- text2text-generation
- text-classification
pretty_name: ChronoCorrect Europeana FR
tags:
- historical-newspapers
- ocr-post-correction
- europeana
- cultural-heritage
- llm-generated
- named-entities
- temporal-expressions
---

# ChronoCorrect-Europeana-FR

ChronoCorrect-Europeana-FR is a derived silver dataset for historical OCR post-correction, built from `biglam/europeana_newspapers`.

Each example contains an OCR paragraph and a conservative post-OCR correction candidate. The dataset also includes JSON-encoded metadata for correction tags, named entities, dates, numbers, automatic risk flags, and human verification placeholders.

## Main fields

- `ocr_text`: original OCR paragraph.
- `corrected_text`: conservative OCR post-correction candidate.
- `correction_policy`: correction policy, currently `conservative`.
- `annotation_status`: silver or silver-needs-review status.
- `llm_annotation_json`: JSON string containing correction tags and semantic change annotations.
- `automatic_risk_flags_json`: JSON string containing automatic risk flags.
- `pre_correction_features_json`: JSON string with entities/dates/numbers before correction.
- `post_correction_features_json`: JSON string with entities/dates/numbers after correction.
- `source_metadata_json`: JSON string with source metadata.

## Note

This is a silver dataset. Corrections and semantic annotations are automatically generated and should be manually verified for gold-standard use.
"""

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to JSONL file generated by build_dataset.py",
    )
    parser.add_argument(
        "--out-dir",
        default="hf_dataset_fixed",
        help="Local output folder for fixed HF dataset",
    )
    parser.add_argument(
        "--hub-dataset-id",
        required=True,
        help="Example: EmanuelaBoros/chronocorrect-europeana-fr",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN"),
        help="Defaults to HF_TOKEN environment variable",
    )
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = build_dataset_dict(
        jsonl_path=args.jsonl,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(dataset)
    print("[features]")
    print(dataset["train"].features)

    print(f"[save] Saving fixed flat dataset to: {args.out_dir}")
    dataset.save_to_disk(args.out_dir)
    write_readme(args.out_dir)

    if args.push:
        if args.hf_token:
            print("[hf] Logging in with token")
            login(token=args.hf_token)
        else:
            print(
                "[hf] No token provided; assuming huggingface-cli login already done."
            )

        print(f"[push] Uploading to: {args.hub_dataset_id}")
        dataset.push_to_hub(
            args.hub_dataset_id,
            private=args.private,
            commit_message="Upload flat-schema ChronoCorrect-Europeana dataset",
        )

        print(f"[done] https://huggingface.co/datasets/{args.hub_dataset_id}")
    else:
        print("[done] Not pushed because --push was not provided.")


if __name__ == "__main__":
    main()
