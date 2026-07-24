"""
Author: Keenan Manpearl
Date: 2024-09-09

Generates data/node_splits.tsv (node, split): split is the run (1-5) in
which a node is held out as test; NO_SPLIT (-1) marks nodes never held out
(still valid training data every fold). Shared across all label types -
see sample_labels.py, which only stores node/label.

    python scripts/generate_splits.py

"""

import pandas as pd

NODE_SPLITS_PATH = "data/node_splits.tsv"
NO_SPLIT = -1


def load_node_splits() -> pd.DataFrame:
    """load node/split for every node"""
    return pd.read_csv(NODE_SPLITS_PATH, sep="\t", index_col="node")


def generate_node_splits() -> None:
    """(Re)build node_splits.tsv from raw_data/sample_breakdown.csv."""
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
