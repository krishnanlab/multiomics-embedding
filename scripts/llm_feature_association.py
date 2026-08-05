"""
Author: Keenan Manpearl
Date: 2026-07-27

Given one microbiome/metabolite feature and a study group, asks Claude
(with web search) to research whether the published literature supports an
association between the two, and returns a standardized JSON verdict:
confidence_score (-1 to 3, literature-only - see the rubric in SYSTEM_PROMPT),
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

Single batch job:
    python scripts/llm_feature_association.py \\
        --feature-list results/permutations_diet_10000/meat_hits.txt \\
        --omics-type metabolite --group infant_12mo_meat \\
        --out-dir results/llm_annotations/

Multiple batch jobs, submitted together as ONE batch so they share a
single system-prompt cache instead of each job writing its own cache entry
(--jobs-file: one job per line, "omics_type,group,feature_list,out_dir"):
    python scripts/llm_feature_association.py --jobs-file jobs.csv
"""

import json
import re
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

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

Score confidence_score using exactly this rubric (integer -1 to 3, not a \
continuous score):
- 3 (Strong): >=2 independent primary studies - ideally in human infants/ \
  children, gut microbiome or metabolome context - directly report this \
  feature (or a clearly named homolog/pathway/compound) associated with this \
  specific group/exposure, in the same direction, with no material \
  contradicting evidence found. Age groups do not need to match exactly: a \
  study population within the same broad developmental stage (e.g. early vs. \
  established complementary feeding, breastfed vs. weaned infants) counts as \
  direct evidence even if its specific age bracket differs from this study's \
  5- or 12-month timepoints.
- 2 (Moderate): exactly 1 direct human study, OR multiple indirect studies \
  (animal models, adult human microbiome, related taxa/pathway-level \
  evidence) pointing the same direction.
- 1 (Weak): only generic/pathway-level or theoretical reasoning found (no \
  study on this specific feature+group combination), or the evidence is \
  mixed/conflicting - but something relevant was found.
- 0 (None found): no relevant literature located via search.
- -1 (Contradicting): the weight of relevant literature found reports this \
  feature associated with the specified group in the OPPOSITE direction from \
  the given hypothesis, without comparably strong supporting evidence for the \
  hypothesized direction. Use this only when at least one study directly \
  addressing this feature+group combination points the opposite way - not \
  for merely weak or absent support (that is 1 or 0).

confidence_score 0 specifically means no evidence was found at all - do not \
use it just because evidence is weak; use 1 for that. confidence_score -1 \
specifically means evidence was found and it contradicts the hypothesized \
direction - do not use it for merely mixed or inconclusive evidence, which \
should be scored 1. Only populate mechanism at confidence_score 2 or 3 - set \
it to null at -1, 0, or 1, since a mechanism is not worth reporting without \
at least moderate support in the hypothesized direction. List every search \
query you actually ran in search_queries_used, for auditability.\
"""

# Uncached: used for isolated single lookups (--feature-id), where a cache
# write would never be read back, since there's no follow-up call to hit it.
SYSTEM_PROMPT_PLAIN = [{"type": "text", "text": SYSTEM_PROMPT}]

# Cached: used for all batch modes. SYSTEM_PROMPT is identical across every
# feature+group+job in a run, so it's cached as a single explicit
# breakpoint. Caching order is tools -> system -> messages, so this
# breakpoint also covers the (equally static) web_search tool definition at
# no extra cost. 1h TTL since a run spanning several jobs/many features can
# easily exceed 5 minutes.
SYSTEM_PROMPT_CACHED = [
    {
        "type": "text",
        "text": SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral", "ttl": "1h"},
    }
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "feature_id": {"type": "string"},
        "omics_type": {"type": "string", "enum": ["microbiome", "metabolite"]},
        "group": {"type": "string"},
        "hypothesis": {"type": "string"},
        "confidence_score": {"type": "integer", "enum": [-1, 0, 1, 2, 3]},
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


def build_prompt(feature_record: dict, group_key: str, cached: bool = True) -> "tuple[list, str]":
    """Build the (system_blocks, user_prompt) pair for one feature+group
    lookup. group_key must be a key of GROUPS. Use cached=False only for
    isolated single lookups (see SYSTEM_PROMPT_PLAIN above)."""
    if group_key not in GROUPS:
        raise ValueError(f"unknown group {group_key!r} - must be one of {sorted(GROUPS)}")
    hypothesis = GROUPS[group_key]
    user_prompt = (
        f"Feature:\n{json.dumps(feature_record, indent=2)}\n\n"
        f"Group under study: {group_key}\n"
        f"Hypothesis to evaluate: this feature is {hypothesis}.\n\n"
        "Research this feature and hypothesis, then respond with the required JSON."
    )
    system_blocks = SYSTEM_PROMPT_CACHED if cached else SYSTEM_PROMPT_PLAIN
    return system_blocks, user_prompt


def query_feature_association(
    client: anthropic.Anthropic,
    feature_record: dict,
    group_key: str,
    model: str = "claude-opus-5",
) -> dict:
    """Single API call: web-search-grounded literature lookup for one
    feature+group, returning a dict matching RESPONSE_SCHEMA. Streams to
    avoid SDK timeouts on the (potentially several-minute) search + adaptive
    thinking turn. Use this for one-off lookups (--feature-id) only; for
    many features across one or more jobs, use build_batch_request()/
    run_batch() below instead."""
    system_blocks, user_prompt = build_prompt(feature_record, group_key, cached=False)

    with client.messages.stream(
        model=model,
        max_tokens=32000,
        thinking={"type": "adaptive"},
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 15}],
        output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        system=system_blocks,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        raise RuntimeError(f"request refused for feature {feature_record['feature_id']!r}: {response.stop_details}")

    text_blocks = [block for block in response.content if block.type == "text"]
    if not text_blocks:
        raise RuntimeError(f"no text response for feature {feature_record['feature_id']!r} (stop_reason={response.stop_reason})")
    return json.loads(text_blocks[-1].text)


def build_batch_request(feature_record: dict, group_key: str, custom_id: str, model: str) -> Request:
    """Build one Message Batches API request for a feature+group lookup.
    Always uses the cached system prompt - batch requests are the case
    caching is meant for."""
    system_blocks, user_prompt = build_prompt(feature_record, group_key, cached=True)
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=32000,
            thinking={"type": "adaptive"},
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 15}],
            output_config={"format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
            system=system_blocks,
            messages=[{"role": "user", "content": user_prompt}],
        ),
    )


def run_batch(client: anthropic.Anthropic, requests: list, poll_interval: int = 30) -> dict:
    """Submit a batch, poll until processing ends, and return
    {custom_id: parsed_json_or_None}. Failures (errored/expired/canceled/
    refused/paused) are logged to stderr and mapped to None so callers can
    tell them apart from successes. Works the same whether `requests` came
    from one job or several - the caller is responsible for routing results
    back to the right job/output directory via custom_id."""
    batch = client.messages.batches.create(requests=requests)
    print(f"submitted batch {batch.id} ({len(requests)} requests)", file=sys.stderr)

    while True:
        batch = client.messages.batches.retrieve(batch.id)
        if batch.processing_status == "ended":
            break
        print(f"batch {batch.id}: {batch.request_counts}", file=sys.stderr)
        time.sleep(poll_interval)

    results: dict = {}
    for entry in client.messages.batches.results(batch.id):
        cid = entry.custom_id
        r = entry.result
        if r.type != "succeeded":
            print(f"FAILED {cid}: {r.type}", file=sys.stderr)
            results[cid] = None
            continue

        message = r.message
        if message.stop_reason == "refusal":
            print(f"FAILED {cid}: refused ({message.stop_details})", file=sys.stderr)
            results[cid] = None
            continue
        if message.stop_reason == "pause_turn":
            # The batch worker's agentic (web-search) loop didn't finish
            # within its allotted iterations. A production version of this
            # script would resubmit the paused assistant content as a
            # follow-up request to continue the turn; not implemented here
            print(f"FAILED {cid}: paused mid-turn, needs continuation", file=sys.stderr)
            results[cid] = None
            continue

        text_blocks = [b for b in message.content if b.type == "text"]
        if not text_blocks:
            print(f"FAILED {cid}: no text content (stop_reason={message.stop_reason})", file=sys.stderr)
            results[cid] = None
            continue
        results[cid] = json.loads(text_blocks[-1].text)

    return results


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _sanitize_filename(feature_id: str) -> str:
    return _SAFE_FILENAME_RE.sub("_", feature_id)

_SAFE_CUSTOM_ID_RE = re.compile(r"[^A-Za-z0-9_-]")

def _sanitize_custom_id(s: str) -> str:
    return _SAFE_CUSTOM_ID_RE.sub("_", s)


def parse_jobs_file(path: str) -> "list[dict]":
    """Parse a jobs manifest: one job per line, formatted
    'omics_type,group,feature_list,out_dir'. Blank lines and lines starting
    with '#' are skipped."""
    jobs = []
    with open(path) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                raise ValueError(f"{path}:{lineno}: expected 4 comma-separated fields, got {len(parts)}: {line!r}")
            omics_type, group, feature_list, out_dir = parts
            if omics_type not in ("microbiome", "metabolite"):
                raise ValueError(f"{path}:{lineno}: omics_type must be 'microbiome' or 'metabolite', got {omics_type!r}")
            if group not in GROUPS:
                raise ValueError(f"{path}:{lineno}: unknown group {group!r} - must be one of {sorted(GROUPS)}")
            jobs.append({"omics_type": omics_type, "group": group, "feature_list": feature_list, "out_dir": out_dir})
    if not jobs:
        raise ValueError(f"{path}: no jobs found")
    return jobs


def run_jobs(client: anthropic.Anthropic, jobs: "list[dict]", model: str, poll_interval: int) -> None:
    """Build one combined batch spanning every job, submit it once, then
    route results back to each job's own out_dir. This is what lets all
    jobs share a single system-prompt cache instead of each job (or each
    separate CLI invocation) writing its own cache entry."""
    dfs: dict = {}  # omics_type -> DataFrame, loaded once per omics_type
    # custom_id -> (job_idx, feature_id, out_dir)
    id_map: dict = {}
    batch_requests: list = []

    for job_idx, job in enumerate(jobs):
        omics_type = job["omics_type"]
        if omics_type not in dfs:
            dfs[omics_type] = load_microbiome_da() if omics_type == "microbiome" else load_metabolite_da()
        df = dfs[omics_type]

        with open(job["feature_list"]) as f:
            feature_ids = [line.strip() for line in f if line.strip()]

        for feature_id in feature_ids:
            try:
                record = build_feature_record(feature_id, omics_type, df)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"FAILED job {job_idx} ({omics_type}/{job['group']}) {feature_id}: {exc}", file=sys.stderr)
                continue
            custom_id = f"{job_idx}-{_sanitize_custom_id(feature_id)}"[:64]
            if custom_id in id_map:
                print(f"FAILED job {job_idx} {feature_id}: custom_id {custom_id!r} collides after truncation, skipping", file=sys.stderr)
                continue
            id_map[custom_id] = (job_idx, feature_id, job["out_dir"])
            batch_requests.append(build_batch_request(record, job["group"], custom_id, model))

    if not batch_requests:
        print("no valid features to submit across any job", file=sys.stderr)
        sys.exit(1)

    print(f"submitting {len(batch_requests)} requests across {len(jobs)} job(s) in one batch", file=sys.stderr)
    results = run_batch(client, batch_requests, poll_interval=poll_interval)

    out_dirs_made: set = set()
    for custom_id, (job_idx, feature_id, out_dir) in id_map.items():
        result = results.get(custom_id)
        if result is None:
            continue
        if out_dir not in out_dirs_made:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            out_dirs_made.add(out_dir)
        out_path = Path(out_dir) / f"{_sanitize_filename(feature_id)}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"job {job_idx} {feature_id}: confidence_score={result['confidence_score']} -> {out_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--feature-id", help="single feature ID to look up")
    target.add_argument("--feature-list", help="path to a file with one feature ID per line (single batch job)")
    target.add_argument("--jobs-file", help="path to a jobs manifest (multiple batch jobs submitted together, one shared cache)")
    parser.add_argument("--omics-type", choices=["microbiome", "metabolite"], help="required with --feature-id or --feature-list")
    parser.add_argument("--group", choices=sorted(GROUPS), help="required with --feature-id or --feature-list")
    parser.add_argument("--out", help="output JSON path (--feature-id mode)")
    parser.add_argument("--out-dir", help="output directory, one JSON file per feature (--feature-list mode)")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--poll-interval", type=int, default=30, help="seconds between batch status checks (batch modes)")
    args = parser.parse_args()

    if args.feature_id:
        if not args.out or not args.omics_type or not args.group:
            parser.error("--feature-id requires --out, --omics-type, and --group")
    elif args.feature_list:
        if not args.out_dir or not args.omics_type or not args.group:
            parser.error("--feature-list requires --out-dir, --omics-type, and --group")

    client = anthropic.Anthropic()

    if args.feature_id:
        df = load_microbiome_da() if args.omics_type == "microbiome" else load_metabolite_da()
        try:
            record = build_feature_record(args.feature_id, args.omics_type, df)
            result = query_feature_association(client, record, args.group, model=args.model)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {args.feature_id}: {exc}", file=sys.stderr)
            raise

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"{args.feature_id}: confidence_score={result['confidence_score']} -> {out_path}")

    elif args.feature_list:
        job = {"omics_type": args.omics_type, "group": args.group, "feature_list": args.feature_list, "out_dir": args.out_dir}
        run_jobs(client, [job], model=args.model, poll_interval=args.poll_interval)

    else:
        jobs = parse_jobs_file(args.jobs_file)
        run_jobs(client, jobs, model=args.model, poll_interval=args.poll_interval)