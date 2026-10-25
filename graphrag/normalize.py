"""I6 normalization: turn surface mentions into stable merge keys.

The display name is kept separately on the node; this module only produces the
identity used for merging and traversal. The suffix whitelist is deliberately
conservative (``motors``, ``group``, ``energy`` are *not* stripped).
"""

from __future__ import annotations

import re
import unicodedata

LEADING_ARTICLES = {"the", "a", "an"}
CORPORATE_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "ltd",
    "limited",
    "llc",
    "plc",
    "gmbh",
    "ag",
    "sa",
    "sas",
    "bv",
    "nv",
    "pty",
    "srl",
    "spa",
    "kk",
    "oy",
}
# Types requested from the extractor. The first six are the upstream spec's type
# set; the medical ones are needed for this corpus and still end in MISC at worst.
ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "PRODUCT",
    "EVENT",
    "MISC",
    "DISEASE",
    "DRUG",
    "PROCEDURE",
    "SYMPTOM",
    "ANATOMY",
    "ORGANISM",
    "GENE",
    "RISK_FACTOR",
)


def normalize_entity(name: str) -> str:
    """Return the canonical merge key for a surface mention."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"['’]s\b", "s", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [token for token in text.split() if token]
    while tokens and tokens[0] in LEADING_ARTICLES:
        tokens.pop(0)
    while len(tokens) > 1 and tokens[-1] in CORPORATE_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def normalize_relation(relation: str) -> str:
    """Return an ``UPPER_SNAKE_CASE`` relation label."""
    text = unicodedata.normalize("NFKD", str(relation))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    for prefix in ("IS_", "ARE_", "WAS_"):
        if text.startswith(prefix) and len(text) > len(prefix):
            remainder = text[len(prefix) :]
            if remainder in {"A", "AN", "THE"}:
                break
            text = remainder
            break
    aliases = {
        "CEO": "CEO_OF",
        "FOUNDER": "FOUNDED_BY",
        "FOUNDED": "FOUNDED_BY",
        "TREATMENT_FOR": "TREATS",
        "TREATMENT_OF": "TREATS",
        "SYMPTOMS_OF": "SYMPTOM_OF",
        "SYMPTOM_FOR": "SYMPTOM_OF",
        "RISK_FOR": "RISK_FACTOR_FOR",
        "DIAGNOSIS_BY": "DIAGNOSED_BY",
        "CAUSE_OF": "CAUSES",
        "CAUSED_BY": "CAUSED_BY",
    }
    return aliases.get(text, text)


def clean_type(value: str) -> str:
    text = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    aliases = {
        "ORGANISATION": "ORGANIZATION",
        "ORG": "ORGANIZATION",
        "COMPANY": "ORGANIZATION",
        "PLACE": "LOCATION",
        "COUNTRY": "LOCATION",
        "CITY": "LOCATION",
        "ILLNESS": "DISEASE",
        "CONDITION": "DISEASE",
        "MEDICATION": "DRUG",
        "MEDICINE": "DRUG",
        "TREATMENT": "PROCEDURE",
        "SYMPTOMS": "SYMPTOM",
        "BODY_PART": "ANATOMY",
        "BACTERIUM": "ORGANISM",
        "VIRUS": "ORGANISM",
        "PATHOGEN": "ORGANISM",
        "RISK": "RISK_FACTOR",
    }
    text = aliases.get(text, text)
    return text if text in ENTITY_TYPES else "MISC"
