"""
Author: Keenan Manpearl
Date: 2026-07-27

Filtering and formatting for
results/differential_abundance/time_{microbiome,metabolites}.txt feature
annotations, ahead of an LLM literature-association lookup
(scripts/llm_feature_association.py). Most rows in both tables are
functionally uninformative ("Function unknown", "<none>", bare chemical
formulas) - filtered out here since asking an LLM to research a literature
association for an unnamed feature just wastes the search. Built on top of
scripts/da_comparison.py's load_baseline_da(), which already handles the
header-skip/encoding quirks of both files.
"""

import re

import pandas as pd

from scripts.da_comparison import load_baseline_da

NAME_COL = "Feature"
FUNCTION_COL = "Feature function/pathway/details"

_UNKNOWN_FUNCTION_RE = re.compile(r"unknown", re.IGNORECASE)
_GENERIC_NAME_RE = re.compile(
    r"^(uncharacterized protein|hypothetical protein|putative protein)$"
    r"|(protein|domain) of unknown function \(duf\d+\)",
    re.IGNORECASE,
)
_TAXONOMY_PREFIX = "k_"

_METABOLITE_FORMULA_RE = re.compile(r"^C\d+ ?H\d+")
_METABOLITE_PLACEHOLDER_RE = re.compile(r"^(<none>|> ?limit)", re.IGNORECASE)

_ID_FIELD_RE = {
    "kegg": re.compile(r"KEGG ID=([^,\]]+)"),
    "cas": re.compile(r"CAS ID=([^,\]]+)"),
    "metlin": re.compile(r"METLIN ID=([^,\]]+)"),
    "lipid": re.compile(r"Lipid ID=([^,\]]+)"),
}

_MICROBIOME_ANNOTATION_TYPES = {
    "K": "KEGG_ortholog",
    "COG": "COG",
    "ENOG": "eggNOG",
    "PF": "Pfam",
}

_METABOLITE_ANNOTATION_TYPES = {
    "N_AQ": "metabolite_aqueous",
    "P_AQ": "metabolite_aqueous",
    "N_LP": "metabolite_lipid",
    "P_LP": "metabolite_lipid",
}


def load_microbiome_da(path: str = "results/differential_abundance/time_microbiome.txt") -> pd.DataFrame:
    """Load results/differential_abundance/time_microbiome.txt via da_comparison.load_baseline_da()."""
    return load_baseline_da(path)


def load_metabolite_da(path: str = "results/differential_abundance/time_metabolites.txt") -> pd.DataFrame:
    """Load results/differential_abundance/time_metabolites.txt via da_comparison.load_baseline_da()."""
    return load_baseline_da(path)


def is_informative_microbiome_row(feature_id: str, row: pd.Series) -> bool:
    """True unless the row's annotation is too uninformative for a literature
    search. Taxonomic lineage rows (feature_id starting with "k_") are always
    informative - the lineage string itself is the identity, even though
    their function/pathway column is NA. Everything else needs both a real
    name and a non-"unknown" function/pathway."""
    if feature_id.startswith(_TAXONOMY_PREFIX):
        return True
    function = row.get(FUNCTION_COL)
    if pd.isna(function) or not str(function).strip() or str(function).strip().upper() == "NA":
        return False
    if _UNKNOWN_FUNCTION_RE.search(str(function)):
        return False
    name = row.get(NAME_COL)
    if pd.isna(name) or not str(name).strip():
        return False
    if _GENERIC_NAME_RE.search(str(name).strip()):
        return False
    return True


def is_informative_metabolite_row(row: pd.Series) -> bool:
    """True unless the row's name is a placeholder ("<none>", "> limit") or a
    bare molecular formula with no compound name - neither is searchable."""
    name = row.get(NAME_COL)
    if pd.isna(name) or not str(name).strip():
        return False
    name = str(name).strip()
    if _METABOLITE_PLACEHOLDER_RE.match(name):
        return False
    if _METABOLITE_FORMULA_RE.match(name):
        return False
    return True


def parse_metabolite_ids(function_text) -> dict:
    """Extract KEGG/CAS/METLIN/Lipid IDs from a metabolites.txt annotation
    blob (e.g. "Feruloylagmatine [ C15 H22 N4 O3, ..., KEGG ID=C18325,
    METLIN ID=7215 ]"). Returns only the keys actually present - a blank
    value (e.g. "CAS ID=,") is treated as absent."""
    if pd.isna(function_text):
        return {}
    text = str(function_text)
    ids = {}
    for key, pattern in _ID_FIELD_RE.items():
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if value:
                ids[key] = value
    return ids


def _microbiome_annotation_type(feature_id: str) -> str:
    if feature_id.startswith(_TAXONOMY_PREFIX):
        return "taxonomy"
    for prefix, label in _MICROBIOME_ANNOTATION_TYPES.items():
        if feature_id.startswith(prefix):
            return label
    raise ValueError(f"unrecognized microbiome feature ID prefix: {feature_id!r}")


def _metabolite_annotation_type(feature_id: str) -> str:
    for prefix, label in _METABOLITE_ANNOTATION_TYPES.items():
        if feature_id.startswith(prefix):
            return label
    raise ValueError(f"unrecognized metabolite feature ID prefix: {feature_id!r}")


_TAXONOMY_RANK_PREFIXES = {
    "phylum": "p_",
    "class": "c_",
    "order": "o_",
    "family": "f_",
    "genus": "g_",
    "species": "s_",
}


def microbiome_taxonomy_rank(feature_id: str, rank: str = "phylum") -> "str | None":
    """Parse one taxonomic rank (default phylum) out of a taxonomy-type
    microbiome feature ID's dot-delimited lineage string (e.g.
    "k_Bacteria.p_Firmicutes.c_Clostridia.o_Eubacteriales...."). Returns None
    for non-taxonomy feature IDs (anything not starting with "k_") - only a
    small minority of microbiome features are taxonomy-typed (~457 of
    17,033), everything else is a functional-annotation ID (COG/eggNOG/Pfam/
    KEGG ortholog) with no lineage to parse. Also returns None if the
    requested rank is simply absent from a given lineage string (e.g. an
    unclassified genus)."""
    if not feature_id.startswith(_TAXONOMY_PREFIX):
        return None
    prefix = _TAXONOMY_RANK_PREFIXES[rank]
    for part in feature_id.split("."):
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def build_feature_record(feature_id: str, omics_type: str, df: pd.DataFrame) -> dict:
    """Build the per-feature JSON block sent to the LLM: feature_id,
    omics_type, annotation_type, name, pathway_or_function (microbiome only),
    external_ids (metabolite only, present keys only). No fold-change/FDR/
    direction is included - confidence is literature-only, so internal study
    statistics never reach the model. Raises KeyError if feature_id isn't in
    df's index."""
    if omics_type not in ("microbiome", "metabolite"):
        raise ValueError(f"omics_type must be 'microbiome' or 'metabolite', got {omics_type!r}")

    row = df.loc[feature_id]
    name = row.get(NAME_COL)
    name = None if pd.isna(name) else str(name).strip()

    record = {"feature_id": feature_id, "omics_type": omics_type}
    if omics_type == "microbiome":
        annotation_type = _microbiome_annotation_type(feature_id)
        record["annotation_type"] = annotation_type
        # taxonomy rows have no NAME_COL value - the feature_id lineage string is the name
        record["name"] = feature_id if annotation_type == "taxonomy" else name
        function = row.get(FUNCTION_COL)
        record["pathway_or_function"] = None if pd.isna(function) else str(function).strip()
    else:
        record["annotation_type"] = _metabolite_annotation_type(feature_id)
        record["name"] = name
        ids = parse_metabolite_ids(row.get(FUNCTION_COL))
        if ids:
            record["external_ids"] = ids
    return record
