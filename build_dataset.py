#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build ChronoCorrect-Europeana from biglam/europeana_newspapers.

Pipeline:
1. Load French subset from biglam/europeana_newspapers.
2. Split long OCR pages into paragraphs.
3. Select paragraphs with names, dates, numbers, or low OCR confidence.
4. Generate conservative OCR correction using OpenAI API.
5. Generate structured semantic annotation using OpenAI API.
6. Run NER/date/number detection before and after correction.
7. Automatically flag risky examples.
8. Write JSONL output for later human verification.

Example with API:

python build_dataset.py \
  --output-jsonl outputs/chronocorrect_europeana_fr_mini.jsonl \
  --max-examples 1000 \
  --max-pages 50000 \
  --language fr \
  --model-correction gpt-5-mini \
  --model-annotation gpt-5-mini \
  --resume \
  --verbose

Dry run without API:

python build_dataset.py \
  --output-jsonl outputs/sample_no_api.jsonl \
  --max-examples 50 \
  --max-pages 500 \
  --no-api \
  --verbose
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional

from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------

try:
    import spacy
except ImportError:
    spacy = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ---------------------------------------------------------------------
# Regexes for dates/numbers
# ---------------------------------------------------------------------

FRENCH_MONTHS = (
    "janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|"
    "septembre|octobre|novembre|décembre|decembre"
)

DATE_PATTERN = re.compile(
    rf"""
    \b(
        \d{{1,2}}\s+(?:{FRENCH_MONTHS})\s+\d{{4}}
        |
        \d{{1,2}}\s+(?:{FRENCH_MONTHS})
        |
        \d{{1,2}}[/-]\d{{1,2}}[/-]\d{{2,4}}
        |
        (?:18|19|20)\d{{2}}
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

NUMBER_PATTERN = re.compile(
    r"""
    \b
    \d+
    (?:[.,]\d+)?
    (?:\s?(?:francs?|millions?|milliards?|kg|kilogr\.?|mètres?|metres?|km|%))?
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------

CORRECTION_SYSTEM_PROMPT = """You are correcting OCR errors in historical French newspaper text.

Correct only errors introduced by OCR.
Preserve historical spelling, wording, abbreviations, punctuation style, and typography when plausible.
Do not modernize the language.
Do not paraphrase.
Do not add missing information.
Do not remove historically meaningful content.
If a word is uncertain, choose the most faithful reading rather than a fluent invention.

Return only the corrected text.
"""

ANNOTATION_SYSTEM_PROMPT = """You compare OCR text and corrected historical French newspaper text.

Identify only changes caused by correction.
Focus on named entities, dates, numbers, places, organizations, titles, and historically meaningful expressions.

Flag possible_hallucination if the correction adds information not supported by the OCR.
Flag possible_overcorrection if the correction modernizes, paraphrases, or rewrites plausible historical wording.
"""


ANNOTATION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "correction_tags": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "CHAR_CONFUSION",
                    "DIACRITIC_RESTORATION",
                    "PUNCTUATION_REPAIR",
                    "WHITESPACE_REPAIR",
                    "LINEBREAK_HYPHENATION",
                    "WORD_SPLIT",
                    "WORD_MERGE",
                    "ENTITY_REPAIR",
                    "DATE_REPAIR",
                    "NUMBER_REPAIR",
                    "ABBREVIATION_REPAIR",
                    "LAYOUT_CONTAMINATION",
                    "BOILERPLATE_REMOVAL",
                    "HISTORICAL_SPELLING_PRESERVED",
                    "UNCERTAIN_READING",
                    "POSSIBLE_HALLUCINATION",
                    "POSSIBLE_OVERCORRECTION",
                ],
            },
        },
        "changed_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ocr_surface": {"type": "string"},
                    "corrected_surface": {"type": "string"},
                    "entity_type": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [
                            "repaired",
                            "preserved",
                            "possibly_changed",
                            "hallucinated",
                            "deleted",
                        ],
                    },
                },
                "required": [
                    "ocr_surface",
                    "corrected_surface",
                    "entity_type",
                    "status",
                ],
            },
        },
        "changed_dates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ocr_surface": {"type": "string"},
                    "corrected_surface": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [
                            "repaired",
                            "preserved",
                            "possibly_changed",
                            "hallucinated",
                            "deleted",
                        ],
                    },
                },
                "required": ["ocr_surface", "corrected_surface", "status"],
            },
        },
        "changed_numbers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "ocr_surface": {"type": "string"},
                    "corrected_surface": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": [
                            "repaired",
                            "preserved",
                            "possibly_changed",
                            "hallucinated",
                            "deleted",
                        ],
                    },
                },
                "required": ["ocr_surface", "corrected_surface", "status"],
            },
        },
        "possible_hallucination": {"type": "boolean"},
        "possible_overcorrection": {"type": "boolean"},
        "uncertain_readings": {
            "type": "array",
            "items": {"type": "string"},
        },
        "short_rationale": {"type": "string"},
    },
    "required": [
        "correction_tags",
        "changed_entities",
        "changed_dates",
        "changed_numbers",
        "possible_hallucination",
        "possible_overcorrection",
        "uncertain_readings",
        "short_rationale",
    ],
}


# ---------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------


@dataclass
class CandidateParagraph:
    source_id: str
    paragraph_id: str
    language: str
    title: Optional[str]
    date: Optional[str]
    mean_ocr: Optional[float]
    std_ocr: Optional[float]
    ocr_text: str
    source_metadata: Dict[str, Any]
    pre_features: Dict[str, Any]
    selection_reasons: List[str]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def get_first_existing(
    example: Dict[str, Any],
    keys: List[str],
    default: Any = None,
) -> Any:
    for key in keys:
        if key in example and example[key] is not None:
            return example[key]
    return default


def normalize_language_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).lower() for v in value]
    return [str(value).lower()]


def is_target_language(example: Dict[str, Any], target_language: str) -> bool:
    values = []
    for key in [
        "language",
        "lang",
        "langs",
        "languages",
        "ocr_lang",
        "detected_language",
    ]:
        values.extend(normalize_language_value(example.get(key)))

    target_language = target_language.lower()

    aliases = {
        "fr": {"fr", "fra", "fre", "french", "français", "francais"},
        "de": {"de", "deu", "ger", "german", "deutsch"},
        "en": {"en", "eng", "english"},
    }

    accepted = aliases.get(target_language, {target_language})
    return any(v in accepted for v in values)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def split_into_paragraphs(
    text: str,
    min_chars: int = 120,
    max_chars: int = 1200,
) -> List[str]:
    if not text:
        return []

    text = text.replace("\r", "\n")

    raw_chunks = re.split(r"\n\s*\n|\n{2,}", text)
    raw_chunks = [re.sub(r"[ \t]+", " ", chunk).strip() for chunk in raw_chunks]
    raw_chunks = [chunk for chunk in raw_chunks if chunk]

    paragraphs: List[str] = []
    buffer = ""

    for chunk in raw_chunks:
        chunk = re.sub(r"\s+", " ", chunk).strip()

        if not chunk:
            continue

        if len(chunk) > max_chars:
            if len(buffer) >= min_chars:
                paragraphs.append(buffer)
                buffer = ""

            sentences = re.split(r"(?<=[.!?;:])\s+", chunk)
            sub_buffer = ""

            for sent in sentences:
                if len(sub_buffer) + len(sent) + 1 <= max_chars:
                    sub_buffer = (sub_buffer + " " + sent).strip()
                else:
                    if len(sub_buffer) >= min_chars:
                        paragraphs.append(sub_buffer)
                    sub_buffer = sent

            if len(sub_buffer) >= min_chars:
                paragraphs.append(sub_buffer)

            continue

        if len(buffer) + len(chunk) + 1 <= max_chars:
            buffer = (buffer + " " + chunk).strip()
        else:
            if len(buffer) >= min_chars:
                paragraphs.append(buffer)
            buffer = chunk

    if len(buffer) >= min_chars:
        paragraphs.append(buffer)

    return paragraphs


def looks_like_bad_paragraph(text: str) -> bool:
    if not text:
        return True

    stripped = text.strip()

    if len(stripped) < 20:
        return True

    alpha = sum(ch.isalpha() for ch in stripped)
    if alpha / max(len(stripped), 1) < 0.45:
        return True

    if stripped.count("|") > 5:
        return True

    if re.search(r"(.)\1{10,}", stripped):
        return True

    return False


def ensure_output_dir(path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)


# ---------------------------------------------------------------------
# NLP detection
# ---------------------------------------------------------------------


def load_spacy_model(model_name: str):
    if spacy is None:
        print(
            "WARNING: spaCy is not installed. Entity detection will be disabled.",
            file=sys.stderr,
        )
        return None

    try:
        tqdm.write(f"[setup] Loading spaCy model: {model_name}")
        return spacy.load(model_name)
    except Exception:
        print(
            f"WARNING: Could not load spaCy model '{model_name}'. "
            f"Install with: python -m spacy download {model_name}",
            file=sys.stderr,
        )
        return None


def detect_entities_dates_numbers(text: str, nlp=None) -> Dict[str, Any]:
    entities = []

    if nlp is not None:
        doc = nlp(text[:3000])
        entities = [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    dates = [m.group(0) for m in DATE_PATTERN.finditer(text)]
    numbers = [m.group(0) for m in NUMBER_PATTERN.finditer(text)]

    return {
        "num_entities": len(entities),
        "entities": entities[:50],
        "num_dates": len(dates),
        "dates": dates[:50],
        "num_numbers": len(numbers),
        "numbers": numbers[:50],
        "has_entity": len(entities) > 0,
        "has_date": len(dates) > 0,
        "has_number": len(numbers) > 0,
    }


def selection_reasons(
    features: Dict[str, Any],
    mean_ocr: Optional[float],
    low_ocr_threshold: float,
) -> List[str]:
    reasons = []

    if mean_ocr is not None and mean_ocr < low_ocr_threshold:
        reasons.append("low_ocr_confidence")

    if features.get("has_entity"):
        reasons.append("contains_named_entity")

    if features.get("has_date"):
        reasons.append("contains_date")

    if features.get("has_number"):
        reasons.append("contains_number")

    return reasons


# ---------------------------------------------------------------------
# OpenAI calls
# ---------------------------------------------------------------------


def make_openai_client():
    if OpenAI is None:
        raise RuntimeError(
            "openai package is not installed. Install with: pip install openai"
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it with: export OPENAI_API_KEY='your_key'"
        )

    return OpenAI()


def call_with_retries(fn, retries: int = 3, sleep_seconds: float = 5.0):
    last_err = None

    for attempt in range(retries):
        try:
            return fn()
        except Exception as err:
            last_err = err
            tqdm.write(f"[retry] Attempt {attempt + 1}/{retries} failed: {repr(err)}")

            if attempt < retries - 1:
                sleep_for = sleep_seconds * (attempt + 1)
                tqdm.write(f"[retry] Sleeping {sleep_for:.1f}s before retry.")
                time.sleep(sleep_for)

    raise last_err


def correct_with_openai(
    client,
    ocr_text: str,
    model: str,
    date: Optional[str] = None,
    title: Optional[str] = None,
    retries: int = 3,
) -> str:
    context_lines = []

    if title:
        context_lines.append(f"Newspaper title: {title}")

    if date:
        context_lines.append(f"Publication date: {date}")

    user_prompt = "\n".join(context_lines)

    if user_prompt:
        user_prompt += "\n\n"

    user_prompt += f"OCR text:\n{ocr_text}"

    def _call():
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": CORRECTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text.strip()

    return call_with_retries(_call, retries=retries)


def annotate_with_openai(
    client,
    ocr_text: str,
    corrected_text: str,
    model: str,
    date: Optional[str] = None,
    title: Optional[str] = None,
    retries: int = 3,
) -> Dict[str, Any]:
    user_prompt = f"""Publication date: {date or "unknown"}
Newspaper title: {title or "unknown"}

OCR text:
{ocr_text}

Corrected text:
{corrected_text}
"""

    def _call():
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "chronocorrect_annotation",
                    "schema": ANNOTATION_JSON_SCHEMA,
                    "strict": True,
                }
            },
        )
        return json.loads(response.output_text)

    return call_with_retries(_call, retries=retries)


# ---------------------------------------------------------------------
# Risk detection
# ---------------------------------------------------------------------


def automatic_risk_flags(
    ocr_text: str,
    corrected_text: str,
    pre_features: Dict[str, Any],
    post_features: Dict[str, Any],
    llm_annotation: Dict[str, Any],
    max_length_ratio: float = 1.35,
) -> Dict[str, Any]:
    flags = {
        "entity_count_changed": False,
        "date_count_changed": False,
        "number_count_changed": False,
        "length_changed_strongly": False,
        "llm_possible_hallucination": False,
        "llm_possible_overcorrection": False,
        "has_uncertain_readings": False,
        "risk_score": 0,
        "risk_reasons": [],
    }

    if pre_features.get("num_entities") != post_features.get("num_entities"):
        flags["entity_count_changed"] = True
        flags["risk_score"] += 2
        flags["risk_reasons"].append("entity_count_changed")

    if pre_features.get("num_dates") != post_features.get("num_dates"):
        flags["date_count_changed"] = True
        flags["risk_score"] += 2
        flags["risk_reasons"].append("date_count_changed")

    if pre_features.get("num_numbers") != post_features.get("num_numbers"):
        flags["number_count_changed"] = True
        flags["risk_score"] += 2
        flags["risk_reasons"].append("number_count_changed")

    if len(corrected_text) > len(ocr_text) * max_length_ratio:
        flags["length_changed_strongly"] = True
        flags["risk_score"] += 1
        flags["risk_reasons"].append("length_changed_strongly")

    if llm_annotation.get("possible_hallucination"):
        flags["llm_possible_hallucination"] = True
        flags["risk_score"] += 3
        flags["risk_reasons"].append("llm_possible_hallucination")

    if llm_annotation.get("possible_overcorrection"):
        flags["llm_possible_overcorrection"] = True
        flags["risk_score"] += 2
        flags["risk_reasons"].append("llm_possible_overcorrection")

    if llm_annotation.get("uncertain_readings"):
        flags["has_uncertain_readings"] = True
        flags["risk_score"] += 1
        flags["risk_reasons"].append("has_uncertain_readings")

    flags["needs_human_verification"] = flags["risk_score"] >= 2

    return flags


# ---------------------------------------------------------------------
# Dataset iteration and candidate creation
# ---------------------------------------------------------------------


def iter_source_examples(
    dataset_name: str,
    split: str,
    streaming: bool,
    trust_remote_code: bool,
) -> Iterable[Dict[str, Any]]:
    return load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )


def make_source_id(example: Dict[str, Any], fallback_index: int) -> str:
    for key in ["id", "identifier", "doc_id", "page_id", "url", "iiif_url"]:
        value = example.get(key)
        if value:
            return str(value)

    text = str(get_first_existing(example, ["text", "ocr_text", "content"], ""))
    return f"source_{fallback_index}_{stable_hash(text)}"


def iter_candidate_paragraphs(
    examples: Iterable[Dict[str, Any]],
    args,
    nlp=None,
) -> Iterator[CandidateParagraph]:
    seen_pages = 0
    skipped_language = 0
    skipped_no_text = 0
    total_paragraphs = 0
    skipped_bad_paragraphs = 0
    skipped_not_selected = 0
    yielded = 0

    scan_pbar = tqdm(
        total=args.max_pages,
        desc="Scanning source pages",
        unit="page",
        leave=True,
    )

    for source_index, example in enumerate(examples):
        if args.max_pages is not None and seen_pages >= args.max_pages:
            break

        scan_pbar.update(1)

        if args.language and not is_target_language(example, args.language):
            skipped_language += 1
            scan_pbar.set_postfix(
                {
                    "kept": yielded,
                    "skip_lang": skipped_language,
                    "paragraphs": total_paragraphs,
                }
            )
            continue

        text = get_first_existing(example, ["text", "ocr_text", "content", "full_text"])

        if not isinstance(text, str) or not text.strip():
            skipped_no_text += 1
            scan_pbar.set_postfix(
                {
                    "kept": yielded,
                    "no_text": skipped_no_text,
                    "paragraphs": total_paragraphs,
                }
            )
            continue

        title = get_first_existing(
            example, ["title", "newspaper_title", "publication_title"]
        )
        date = get_first_existing(example, ["date", "publication_date", "year"])
        mean_ocr = safe_float(
            get_first_existing(example, ["mean_ocr", "ocr_mean", "ocr_confidence"])
        )
        std_ocr = safe_float(get_first_existing(example, ["std_ocr", "ocr_std"]))

        source_id = make_source_id(example, source_index)

        paragraphs = split_into_paragraphs(
            text,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )

        seen_pages += 1
        total_paragraphs += len(paragraphs)

        for para_index, paragraph in enumerate(paragraphs):
            if looks_like_bad_paragraph(paragraph):
                skipped_bad_paragraphs += 1
                continue

            features = detect_entities_dates_numbers(paragraph, nlp=nlp)
            reasons = selection_reasons(features, mean_ocr, args.low_ocr_threshold)

            if args.select_all:
                reasons = reasons or ["selected_by_select_all"]

            if not reasons:
                skipped_not_selected += 1
                continue

            metadata = {}

            for key in [
                "url",
                "iiif_url",
                "rights",
                "provider",
                "country",
                "language",
                "lang",
                "title",
                "date",
                "mean_ocr",
                "std_ocr",
                "bounding_boxes",
            ]:
                if key in example:
                    value = example[key]

                    if key == "bounding_boxes":
                        metadata[key] = "present"
                    else:
                        metadata[key] = value

            paragraph_id = f"{source_id}_p{para_index}_{stable_hash(paragraph)}"
            yielded += 1

            scan_pbar.set_postfix(
                {
                    "kept": yielded,
                    "paragraphs": total_paragraphs,
                    "bad": skipped_bad_paragraphs,
                    "not_selected": skipped_not_selected,
                }
            )

            yield CandidateParagraph(
                source_id=source_id,
                paragraph_id=paragraph_id,
                language=args.language,
                title=title,
                date=str(date) if date is not None else None,
                mean_ocr=mean_ocr,
                std_ocr=std_ocr,
                ocr_text=paragraph,
                source_metadata=metadata,
                pre_features=features,
                selection_reasons=reasons,
            )

    scan_pbar.close()

    tqdm.write("Source scanning finished.")
    tqdm.write(f"Pages seen: {seen_pages}")
    tqdm.write(f"Candidates yielded: {yielded}")
    tqdm.write(f"Paragraphs created: {total_paragraphs}")
    tqdm.write(f"Skipped language: {skipped_language}")
    tqdm.write(f"Skipped no text: {skipped_no_text}")
    tqdm.write(f"Skipped bad paragraphs: {skipped_bad_paragraphs}")
    tqdm.write(f"Skipped not selected: {skipped_not_selected}")


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------


def load_existing_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()

    ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                ids.add(obj.get("id"))
            except Exception:
                continue

    return ids


def write_jsonl_record(path: str, record: Dict[str, Any]) -> None:
    ensure_output_dir(path)

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ChronoCorrect-Europeana from biglam/europeana_newspapers."
    )

    parser.add_argument(
        "--dataset-name",
        default="biglam/europeana_newspapers",
        help="HF dataset name.",
    )

    parser.add_argument(
        "--split",
        default="train",
        help="HF split to load.",
    )

    parser.add_argument(
        "--language",
        default="fr",
        help="Target language code, e.g. fr.",
    )

    parser.add_argument(
        "--output-jsonl",
        required=True,
        help="Output JSONL file.",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum source pages/documents to scan.",
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="Maximum selected paragraph examples to output.",
    )

    parser.add_argument(
        "--min-chars",
        type=int,
        default=120,
        help="Minimum paragraph length.",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=1200,
        help="Maximum paragraph length.",
    )

    parser.add_argument(
        "--low-ocr-threshold",
        type=float,
        default=0.80,
        help="Mean OCR confidence threshold for low OCR selection.",
    )

    parser.add_argument(
        "--spacy-model",
        default="fr_core_news_md",
        help="spaCy model for NER.",
    )

    parser.add_argument(
        "--model-correction",
        default="gpt-5-mini",
        help="OpenAI model for correction.",
    )

    parser.add_argument(
        "--model-annotation",
        default="gpt-5-mini",
        help="OpenAI model for structured annotation.",
    )

    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Do not call OpenAI API. Use OCR text as corrected_text placeholder.",
    )

    parser.add_argument(
        "--select-all",
        action="store_true",
        help="Select all valid paragraphs, not only semantic/low-OCR ones.",
    )

    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="Use HF streaming mode.",
    )

    parser.add_argument(
        "--no-streaming",
        action="store_false",
        dest="streaming",
        help="Disable HF streaming.",
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to load_dataset.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output JSONL, skipping existing IDs.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between API examples.",
    )

    parser.add_argument(
        "--api-retries",
        type=int,
        default=3,
        help="Number of retries for API calls.",
    )

    parser.add_argument(
        "--gold-candidate-risk-threshold",
        type=int,
        default=2,
        help="Risk score threshold for needs_human_verification.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra progress messages.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    ensure_output_dir(args.output_jsonl)

    tqdm.write("[setup] Starting ChronoCorrect-Europeana builder")
    tqdm.write(f"[setup] Dataset: {args.dataset_name}")
    tqdm.write(f"[setup] Split: {args.split}")
    tqdm.write(f"[setup] Language: {args.language}")
    tqdm.write(f"[setup] Output: {args.output_jsonl}")
    tqdm.write(f"[setup] Max examples: {args.max_examples}")
    tqdm.write(f"[setup] Max pages: {args.max_pages}")
    tqdm.write(f"[setup] No API mode: {args.no_api}")

    nlp = load_spacy_model(args.spacy_model)

    client = None
    if not args.no_api:
        tqdm.write("[setup] Creating OpenAI client")
        client = make_openai_client()
    else:
        tqdm.write("[setup] Skipping OpenAI client because --no-api is enabled")

    existing_ids = load_existing_ids(args.output_jsonl) if args.resume else set()

    if args.resume:
        tqdm.write(f"[setup] Resume enabled. Existing IDs loaded: {len(existing_ids)}")

    tqdm.write("[setup] Loading Hugging Face dataset stream")

    source_examples = iter_source_examples(
        dataset_name=args.dataset_name,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=args.trust_remote_code,
    )

    tqdm.write("[setup] Creating candidate paragraph iterator")

    candidates = iter_candidate_paragraphs(source_examples, args, nlp=nlp)

    written = 0
    skipped_existing = 0

    step_stats = {
        "seen_candidates": 0,
        "skipped_existing": 0,
        "corrected": 0,
        "annotated": 0,
        "risk_flagged": 0,
        "written": 0,
        "api_errors": 0,
    }

    pbar = tqdm(
        total=args.max_examples,
        desc="Writing ChronoCorrect records",
        unit="example",
        leave=True,
    )

    for candidate in candidates:
        if written >= args.max_examples:
            break

        step_stats["seen_candidates"] += 1

        record_id = (
            f"chronocorrect_europeana_{candidate.language}_"
            f"{stable_hash(candidate.paragraph_id)}"
        )

        if args.verbose:
            tqdm.write(
                f"[candidate] {step_stats['seen_candidates']} | "
                f"id={candidate.paragraph_id} | "
                f"chars={len(candidate.ocr_text)} | "
                f"reasons={candidate.selection_reasons}"
            )

        pbar.set_postfix(
            {
                "stage": "candidate",
                "seen": step_stats["seen_candidates"],
                "written": written,
                "risk": step_stats["risk_flagged"],
            }
        )

        if record_id in existing_ids:
            skipped_existing += 1
            step_stats["skipped_existing"] += 1

            pbar.set_postfix(
                {
                    "stage": "skip_existing",
                    "skipped": step_stats["skipped_existing"],
                    "written": written,
                }
            )
            continue

        if args.no_api:
            pbar.set_postfix({"stage": "no_api_placeholder", "written": written})

            corrected_text = candidate.ocr_text

            llm_annotation = {
                "correction_tags": [],
                "changed_entities": [],
                "changed_dates": [],
                "changed_numbers": [],
                "possible_hallucination": False,
                "possible_overcorrection": False,
                "uncertain_readings": [],
                "short_rationale": "No API mode: correction and annotation not generated.",
            }

        else:
            try:
                pbar.set_postfix(
                    {
                        "stage": "gpt_correction",
                        "written": written,
                        "chars": len(candidate.ocr_text),
                    }
                )

                corrected_text = correct_with_openai(
                    client=client,
                    ocr_text=candidate.ocr_text,
                    model=args.model_correction,
                    date=candidate.date,
                    title=candidate.title,
                    retries=args.api_retries,
                )

                step_stats["corrected"] += 1

                pbar.set_postfix(
                    {
                        "stage": "gpt_annotation",
                        "written": written,
                        "corrected": step_stats["corrected"],
                    }
                )

                llm_annotation = annotate_with_openai(
                    client=client,
                    ocr_text=candidate.ocr_text,
                    corrected_text=corrected_text,
                    model=args.model_annotation,
                    date=candidate.date,
                    title=candidate.title,
                    retries=args.api_retries,
                )

                step_stats["annotated"] += 1

            except Exception as err:
                step_stats["api_errors"] += 1
                tqdm.write(f"[API ERROR] {record_id}: {repr(err)}")

                if args.resume:
                    tqdm.write("[INFO] Continuing because --resume is enabled.")
                    continue

                raise

        pbar.set_postfix({"stage": "post_ner", "written": written})

        post_features = detect_entities_dates_numbers(corrected_text, nlp=nlp)

        pbar.set_postfix({"stage": "risk_flags", "written": written})

        risk_flags = automatic_risk_flags(
            ocr_text=candidate.ocr_text,
            corrected_text=corrected_text,
            pre_features=candidate.pre_features,
            post_features=post_features,
            llm_annotation=llm_annotation,
        )

        risk_flags["needs_human_verification"] = (
            risk_flags["risk_score"] >= args.gold_candidate_risk_threshold
        )

        if risk_flags["needs_human_verification"]:
            step_stats["risk_flagged"] += 1

        pbar.set_postfix(
            {
                "stage": "writing",
                "written": written,
                "risk": step_stats["risk_flagged"],
            }
        )

        record = {
            "id": record_id,
            "source_dataset": args.dataset_name,
            "source_id": candidate.source_id,
            "paragraph_id": candidate.paragraph_id,
            "language": candidate.language,
            "title": candidate.title,
            "date": candidate.date,
            "mean_ocr": candidate.mean_ocr,
            "std_ocr": candidate.std_ocr,
            "ocr_text": candidate.ocr_text,
            "corrected_text": corrected_text,
            "correction_policy": "conservative",
            "selection_reasons": candidate.selection_reasons,
            "pre_correction_features": candidate.pre_features,
            "post_correction_features": post_features,
            "llm_annotation": llm_annotation,
            "automatic_risk_flags": risk_flags,
            "annotation_status": (
                "silver"
                if not risk_flags["needs_human_verification"]
                else "silver_needs_review"
            ),
            "human_verification": {
                "faithful": None,
                "historical_language_preserved": None,
                "entities_correct": None,
                "dates_numbers_correct": None,
                "hallucination_present": None,
                "overcorrection_present": None,
                "notes": None,
            },
            "source_metadata": candidate.source_metadata,
        }

        write_jsonl_record(args.output_jsonl, record)

        written += 1
        step_stats["written"] = written

        pbar.update(1)

        pbar.set_postfix(
            {
                "stage": "done_record",
                "written": written,
                "risk": step_stats["risk_flagged"],
                "api_errors": step_stats["api_errors"],
            }
        )

        if args.sleep > 0:
            pbar.set_postfix({"stage": f"sleep_{args.sleep}s", "written": written})
            time.sleep(args.sleep)

    pbar.close()

    print("\nDone.")
    print(f"Wrote: {written}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Output: {args.output_jsonl}")

    print("\nStep statistics:")
    for key, value in step_stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
