"""
Author: Keenan Manpearl
Date: 2024-09-09

Study-specific label generation and loading, shared by train_deployment_models.py,
train_baseline_models.py, and model.py.

Run this script directly to (re)generate data/time_labels.tsv and
data/diet_labels.tsv from all_data/raw/sample_breakdown.csv and
all_data/raw/microbiome_info_data.csv:

    python scripts/sample_labels.py

Each tsv has just two columns: node (index) and label (binary 0/1). CV fold
assignment is stored separately in data/node_splits.tsv (see
generate_splits.py) and shared across all label types - Dataset.from_label_tsv
joins the two on node.

"""

import pandas as pd

TIME_LABELS_PATH = "data/time_labels.tsv"
DIET_LABELS_PATH = "data/diet_labels.tsv"


def load_time_labels() -> pd.DataFrame:
    """load node/label for the time-point classification target"""
    return pd.read_csv(TIME_LABELS_PATH, sep="\t", index_col="node")


def load_diet_labels() -> pd.DataFrame:
    """load node/label for the diet classification target"""
    return pd.read_csv(DIET_LABELS_PATH, sep="\t", index_col="node")


def generate_label_tsvs() -> None:
    """
    (Re)build time_labels.tsv and diet_labels.tsv from the raw study data.
    """
    breakdown = pd.read_csv("all_data/raw/sample_breakdown.csv")
    breakdown_by_node = breakdown.drop_duplicates("nodes").set_index("nodes")

    time_labels = breakdown_by_node["Time"].map({"Baseline": 0, "Endpoint": 1})
    time_labels = time_labels.rename("label").rename_axis("node").to_frame()
    time_labels.to_csv(TIME_LABELS_PATH, sep="\t")

    diet_info = pd.read_csv(
        "all_data/raw/microbiome_info_data.csv",
        usecols=["sample.name", "Group", "Time"],
    )
    diet_info = diet_info.set_index("sample.name")
    diet_info = diet_info[~diet_info.index.duplicated(keep="first")]

    diet_nodes = time_labels.index[time_labels.index.str.contains("End")]
    diet_labels = diet_info.loc[diet_nodes, "Group"].map({"Dairy": 0, "Meat": 1})
    diet_labels = diet_labels.rename("label").rename_axis("node").to_frame()
    diet_labels.to_csv(DIET_LABELS_PATH, sep="\t")


if __name__ == "__main__":
    generate_label_tsvs()
