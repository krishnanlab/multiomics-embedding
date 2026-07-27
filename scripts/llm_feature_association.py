"""
Author: Keenan Manpearl
Date: 2026-07-27

Given one microbiome/metabolite feature and a study group, asks Claude
(with web search) to research whether the published literature supports an
association between the two, and returns a standardized JSON verdict:
confidence_score (0-3, literature-only - see the rubric in SYSTEM_PROMPT),
a possible mechanism (only populated at confidence_score 2-3), and the
supporting citations found via search. Confidence is deliberately
independent of this study's own statistics (fold change/FDR/consensus
score) - those are only used upstream, outside this script, to decide which
features are worth asking about; see scripts/feature_annotation.py for the
feature-selection/filtering side of that.

Single lookup:
    python scripts/llm_feature_association.py \\
        --feature-id K10189 --omics-type microbiome \\
        --group infant_12mo --out results/llm_annotations/K10189.json

Batch (one feature ID per line in --feature-list):
    python scripts/llm_feature_association.py \\
        --feature-list results/permutations_diet_10000/meat_hits.txt \\
        --omics-type metabolite --group infant_12mo_meat \\
        --out-dir results/llm_annotations/
"""

import json
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic

from scripts.feature_annotation import build_feature_record, load_metabolite_da, load_microbiome_da

GROUPS = {
    "infant_5mo": "increased or characteristic in 5-month-old infants, relative to 12-month-old infants",
    "infant_12mo": "increased or characteristic in 12-month-old infants, relative to 5-month-old infants",
    "infant_12mo_meat": (
        "increased or characteristic in 12-month-old infants receiving meat-based complementary "
        "protein, relative to those receiving dairy-based complementary protein"
    ),
    "infant_12mo_dairy": (
        "increased or characteristic in 12-month-old infants receiving dairy-based complementary "
        "protein, relative to those receiving meat-based complementary protein"
    ),
}

SYSTEM_PROMPT = """\
You are a biomedical literature research assistant helping investigators on an \
infant gut microbiome/metabolome study evaluate candidate microbe/metabolite \
features. For each request you are given one feature and one directional \
hypothesis about an infant study group. Your job is to search the published \
literature and judge how well-established that specific association is - \
independent of any internal study data, which you are not given and must not \
assume.

Use the web_search tool. Formulate multiple search queries (the feature's \
name, any external database IDs given, plus the group's keywords - e.g. \
"infant gut microbiome", "complementary feeding", "weaning", "dietary protein \
source") rather than relying on a single query. Prefer primary research \
articles over reviews when judging directness of evidence, but reviews are \
useful for corroboration.

Cite ONLY sources you actually observed in your web_search results this turn \
- never a citation recalled from training data. Every evidence entry must \
have a url from an actual search result; include a doi only if it was visible \
in that result (otherwise leave it null).

Score confidence_score using exactly this rubric (integer 0-3, not a \
continuous score):
- 3 (Strong): >=2 independent primary studies - ideally in human infants/ \
  children, gut microbiome or metabolome context - directly report this \
  feature (or a clearly named homolog/pathway/compound) associated with this \
  specific group/exposure, in the same direction, with no material \
  contradicting evidence found.
- 2 (Moderate): exactly 1 direct human study, OR multiple indirect studies \
  (animal models, adult human microbiome, related taxa/pathway-level \
  evidence) pointing the same direction.
- 1 (Weak): only generic/pathway-level or theoretical reasoning found (no \
  study on this specific feature+group combination), or the evidence is \
  mixed/conflicting - but something relevant was found.
- 0 (None found): no relevant literature located via search.

confidence_score 0 specifically means no evidence was found at all - do not \
use it just because evidence is weak; use 1 for that. Only populate \
mechanism at confidence_score 2 or 3 - set it to null at 0 or 1, since a \
mechanism is not worth reporting without at least moderate support. List \
every search query you actually ran in search_queries_used, for auditability.\
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "feature_id": {"type": "string"},
        "omics_type": {"type": "string", "enum": ["microbiome", "metabolite"]},
        "group": {"type": "string"},
        "hypothesis": {"type": "string"},
        "confidence_score": {"type": "integer", "enum": [0, 1, 2, 3]},
        "rationale": {"type": "string"},
        "mechanism": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "doi": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "title": {"type": "string"},
                    "study_population": {"type": "string"},
                    "relevance": {"type": "string"},
                },
                "required": ["url", "doi", "title", "study_population", "relevance"],
                "additionalProperties": False,
            },
        },
        "contradicting_evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "search_queries_used": {"type": "array", "items": {"type": "string"}},
        "caveats": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": [
        "feature_id",
        "omics_type",
        "group",
        "hypothesis",
        "confidence_score",
        "rationale",
        "mechanism",
        "evidence",
        "contradicting_evidence",
        "search_queries_used",
        "caveats",
    ],
    "additionalProperties": False,
}


def build_prompt(feature_record: dict, group_key: str) -> "tuple[str, str]":
    """Build the (system, user) prompt pair for one feature+group lookup.
    group_key must be a key of GROUPS."""
    if group_key not in GROUPS:
        raise ValueError(f"unknown group {group_key!r} - must be one of {sorted(GROUPS)}")
    hypothesis = GROUPS[group_key]
    user_prompt = (
        f"Feature:\n{json.dumps(feature_record, indent=2)}\n\n"
        f"Group under study: {group_key}\n"
        f"Hypothesis to evaluate: this feature is {hypothesis}.\n\n"
        "Research this feature and hypothesis, then respond with the required JSON."
    )
    return SYSTEM_PROMPT, user_prompt


def query_feature_association(
    client: anthropic.Anthropic,
    feature_record: dict,
    group_key: str,
    model: str = "claude-opus-5",
) -> dict:
    """Single API call: web-search-grounded literature lookup for one
    feature+group, returning a dict matching RESPONSE_SCHEMA. Streams to
    avoid SDK timeouts on the (potentially several-minute) search + adaptive
    thinking turn."""
    system_prompt, user_prompt = build_prompt(feature_record, group_key)

    with client.messages.stream(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 8}],
        output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError(f"request refused for feature {feature_record['feature_id']!r}: {response.stop_details}")

    text_blocks = [block for block in response.content if block.type == "text"]
    if not text_blocks:
        raise RuntimeError(f"no text response for feature {feature_record['feature_id']!r} (stop_reason={response.stop_reason})")
    return json.loads(text_blocks[-1].text)


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(feature_id: str) -> str:
    return _SAFE_FILENAME_RE.sub("_", feature_id)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--feature-id", help="single feature ID to look up")
    target.add_argument("--feature-list", help="path to a file with one feature ID per line (batch mode)")
    parser.add_argument("--omics-type", required=True, choices=["microbiome", "metabolite"])
    parser.add_argument("--group", required=True, choices=sorted(GROUPS))
    parser.add_argument("--out", help="output JSON path (single-lookup mode)")
    parser.add_argument("--out-dir", help="output directory, one JSON file per feature (batch mode)")
    parser.add_argument("--model", default="claude-opus-5")
    args = parser.parse_args()

    if args.feature_id and not args.out:
        parser.error("--out is required with --feature-id")
    if args.feature_list and not args.out_dir:
        parser.error("--out-dir is required with --feature-list")

    df = load_microbiome_da() if args.omics_type == "microbiome" else load_metabolite_da()
    client = anthropic.Anthropic()

    if args.feature_id:
        feature_ids = [args.feature_id]
    else:
        with open(args.feature_list) as f:
            feature_ids = [line.strip() for line in f if line.strip()]

    for feature_id in feature_ids:
        try:
            record = build_feature_record(feature_id, args.omics_type, df)
            result = query_feature_association(client, record, args.group, model=args.model)
        except Exception as exc:  # noqa: BLE001 - report and continue in batch mode
            print(f"FAILED {feature_id}: {exc}", file=sys.stderr)
            if args.out:
                raise
            continue

        out_path = Path(args.out) if args.out else Path(args.out_dir) / f"{_sanitize_filename(feature_id)}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"{feature_id}: confidence_score={result['confidence_score']} -> {out_path}")
