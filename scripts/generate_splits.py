"""
Author: Keenan Manpearl
Date: 2024-09-09

Generates data/node_splits.tsv - a two-column (node, split) tsv replacing
the old multi-column raw_data/sample_breakdown.csv for CV purposes.
split is the run (1-5) in which a node is held out as test; NO_SPLIT (-1)
marks nodes that are never held out (still valid training data for every
fold, just never evaluated on). This file is shared by every label type
(time, diet, ...) - see sample_labels.py, which only stores node/label.

Run this script directly to (re)generate data/node_splits.tsv:

    python scripts/generate_splits.py

"""

import pandas as pd

NODE_SPLITS_PATH = "data/node_splits.tsv"
NO_SPLIT = -1


def load_node_splits() -> pd.DataFrame:
    """load node/split for every node"""
    return pd.read_csv(NODE_SPLITS_PATH, sep="\t", index_col="node")


def generate_node_splits() -> None:
    """
    (Re)build node_splits.tsv from raw_data/sample_breakdown.csv.
    split = the run in which a node is held out as test (NO_SPLIT if it's
    never held out in any run).
    """
    breakdown = pd.read_csv("raw_data/sample_breakdown.csv")
    all_nodes = breakdown["nodes"].drop_duplicates()

    split = (
        breakdown[breakdown["partition"] == "test"]
        .drop_duplicates("nodes")
        .set_index("nodes")["run"]
    )
    split = split.reindex(all_nodes, fill_value=NO_SPLIT)

    node_splits = split.rename("split").rename_axis("node").to_frame()
    node_splits.to_csv(NODE_SPLITS_PATH, sep="\t")


if __name__ == "__main__":
    generate_node_splits()
