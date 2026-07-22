"""
Author: Keenan Manpearl
Date: 2026-07-21

Combines the raw microbiome and metabolite differential-abundance tables
into one sample x feature matrix, rank-normalizes each feature, and writes
the resulting sample-feature graph as an edge list for node2vec+ embedding.

Sample IDs are mapped to the "<Vial.ID>_<Time>" convention (e.g.
"MD-02_Base") used by
raw_data/microbe_metabolites_filtered_rank_normalized.csv and
data/nodes/samples.txt: metabolite rows already carry Vial.ID/Time,
microbiome rows are joined to a sample name via
raw_data/microbiome_info_data.csv's Library -> sample.name crosswalk.
Feature (column) names are left exactly as they appear in the raw files,
including the long taxonomy strings used for species-level microbiome
columns

Rank normalization: for each feature independently, values are converted to
a fractional rank - rank(x, ties="average") / count(non-missing x), in
(0, 1] - computed on that feature's full native table (all raw microbiome
rows / all raw metabolite rows) *before* restricting to the sample overlap
between the two omics. A raw value of exactly 0 (not detected) still
counts as a real, rankable value for this computation - it affects the
rank denominator and other values' relative ranks the same as any other
measurement.

A true missing value (the feature was never measured for that sample -
not the same as a measured 0) is never ranked at all, and never becomes
an edge.

edges.tsv is a sample-feature edge list additionally excluding any cell
whose *raw* value was exactly 0, even though it still had a real rank
score.

Also writes data/nodes/samples.txt, data/nodes/microbes.txt, and
data/nodes/metabolites.txt from this same merged feature set, since
they're used elsewhere (e.g. src/validation.py's node-list checks,
src/zscoring.py) as the authoritative sample/feature-type node lists
A feature counts as a metabolite if its column name starts with "N_" or "P_",
everything else (COG/ENOG/PF/K codes and taxonomy strings) is a microbial feature.

Run this script directly to (re)generate data/edges.tsv and
data/nodes/{samples,microbes,metabolites}.txt:
"""

import pandas as pd

RAW_DIR = "raw_data"
EDGES_PATH = "data/edges.tsv"
SAMPLES_PATH = "data/nodes/samples.txt"
MICROBES_PATH = "data/nodes/microbes.txt"
METABOLITES_PATH = "data/nodes/metabolites.txt"

METABOLITE_META_COLS = [
    "Group",
    "Time",
    "ms.vial",
    "Vial.ID",
    "count",
    "batch",
    "WAZ",
    "WLZ",
    "LAZ",
    "HCAZ",
]
MICROBIOME_META_COLS = [
    "Library",
    "ID",
    "Timepoint",
    "WAZ",
    "WLZ",
    "LAZ",
    "HCZ",
    "Group",
    "Time",
    "READS",
]


def _rank_normalize(features: pd.DataFrame) -> pd.DataFrame:
    """
    fractional rank (rank / count of non-missing) for each column
    independently - 0 counts as a real, rankable value here; see
    build_edge_list for where 0-valued cells are excluded from the graph
    despite that - they still get a real rank score in this computation
    """
    # average so features with the same rank have same score
    return features.rank(method="average") / features.count()


def load_metabolite_features() -> pd.DataFrame:
    """load metabolite_data_for_differential_abundance.csv, indexed by sample ID"""
    metab = pd.read_csv(f"{RAW_DIR}/metabolite_data_for_differential_abundance.csv")
    metab["sample"] = metab["Vial.ID"] + "_" + metab["Time"]
    return metab.drop(columns=METABOLITE_META_COLS).set_index("sample")


def load_microbiome_features() -> pd.DataFrame:
    """load microbiome_data_for_differential_abundance.csv, indexed by sample ID"""
    micro = pd.read_csv(
        f"{RAW_DIR}/microbiome_data_for_differential_abundance.csv", low_memory=False
    )
    info = pd.read_csv(f"{RAW_DIR}/microbiome_info_data.csv")
    library_to_sample = info.set_index("Library")["sample.name"]
    micro["sample"] = micro["Library"].map(library_to_sample)
    return micro.drop(columns=MICROBIOME_META_COLS).set_index("sample")


def build_merged_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and rank-normalize microbiome/metabolite features separately (each
    on its own full sample set), then inner-join the raw features and the
    rank-normalized features down to the samples present in both omics.
    Returns (merged_raw, merged_ranked).
    """
    metab_features = load_metabolite_features()
    micro_features = load_microbiome_features()

    metab_ranked = _rank_normalize(metab_features)
    micro_ranked = _rank_normalize(micro_features)

    merged_raw = micro_features.join(metab_features, how="inner")
    merged_ranked = micro_ranked.join(metab_ranked, how="inner")
    return merged_raw, merged_ranked


def build_edge_list(raw: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    """
    melt a sample x feature rank-normalized matrix into a sample/feature/weight
    edge list. A true missing value (never measured) is always dropped,
    never filled with a 0 weight. A cell whose *raw* value was exactly 0 is
    also dropped, even though ranked already gave it a real (small,
    non-zero) score.
    """
    edges = ranked.reset_index().melt(
        id_vars="sample", var_name="feature", value_name="weight"
    )
    raw_values = raw.reset_index().melt(
        id_vars="sample", var_name="feature", value_name="raw_value"
    )["raw_value"]
    # raw_value != 0 keeps true-missing (NaN) rows too - NaN != 0 is
    # True - only exact-0 raw values get dropped here
    edges = edges[raw_values != 0]
    return edges.dropna(subset=["weight"])


def is_metabolite_feature(column: str) -> bool:
    """metabolite feature columns are tagged with an ion-mode prefix, N_ or P_"""
    return column.startswith("N_") or column.startswith("P_")


def write_node_lists(columns: "pd.Index[str]", samples: "pd.Index[str]") -> None:
    """(re)write data/nodes/{samples,microbes,metabolites}.txt from the merged data"""
    microbes = sorted(c for c in columns if not is_metabolite_feature(c))
    metabolites = sorted(c for c in columns if is_metabolite_feature(c))
    for path, node_ids in [
        (SAMPLES_PATH, sorted(samples)),
        (MICROBES_PATH, microbes),
        (METABOLITES_PATH, metabolites),
    ]:
        with open(path, "w") as f:
            f.writelines(f"{node_id}\n" for node_id in node_ids)


if __name__ == "__main__":
    merged_raw, merged_ranked = build_merged_features()
    write_node_lists(merged_raw.columns, merged_raw.index)

    edges = build_edge_list(merged_raw, merged_ranked)
    edges.to_csv(EDGES_PATH, sep="\t", header=False, index=False)
