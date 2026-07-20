"""
Author: Keenan Manpearl
Date: 2024-09-09

Study-specific label loaders shared by train_deployment_models.py,
train_baseline_models.py, and model.py: reads this study's
data/raw/sample_breakdown.csv (partition/run/nodes columns) and
data/raw/microbiome_info_data.csv (Group/Time columns, Dairy/Meat mapping)
to build binary time-point and diet labels. Not part of the generic src/
library - this is where this dataset's specific columns/paths live.

"""

import pandas as pd
import numpy as np


def load_test_indices() -> pd.DataFrame:
    """
    Load the test indices for cross validation
    """
    labels = pd.read_csv("data/raw/sample_breakdown.csv")
    subset = labels[labels["partition"] == "test"]
    return subset[["nodes", "run"]]


def get_diet_indices(time_indices: pd.DataFrame) -> pd.DataFrame:
    """
    Get the endpoint samples for diet classification
    """
    return time_indices.loc[time_indices["nodes"].str.contains("End")]


def create_timepoint_labels(time_index: pd.Index) -> pd.Series:
    """
    Convert time points into binary labels for ML
    """
    return pd.Series(
        [1 if "End" in item else 0 for item in time_index], index=time_index
    )


def load_diet_labels(diet_index: pd.Index) -> pd.Series:
    """
    Convert diet into binary labels for ML
    """
    labels = pd.read_csv(
        "data/raw/microbiome_info_data.csv", usecols=["sample.name", "Group", "Time"]
    )
    labels.index = labels["sample.name"]
    labels["Label"] = labels["Group"].map({"Dairy": 0, "Meat": 1})
    labels = labels[~labels.index.duplicated(keep="first")]
    return labels.loc[diet_index, "Label"]


def load_timepoint_labels(model: int) -> pd.DataFrame:
    """
    Convert time points into binary labels for ML, for one sweep fold
    """
    labels = pd.read_csv("data/raw/sample_breakdown.csv")
    labels = labels[labels["run"] == model]
    labels["Label"] = labels["Time"].map({"Baseline": 0, "Endpoint": 1})
    labels.index = labels["nodes"]
    return labels


def load_sweep_diet_labels() -> pd.DataFrame:
    """
    Convert diet into binary labels for ML, for the sweep pipeline
    """
    labels = pd.read_csv(
        "data/raw/microbiome_info_data.csv", usecols=["sample.name", "Group", "Time"]
    )
    labels.index = labels["sample.name"]
    labels["Label"] = labels["Group"].map({"Dairy": 0, "Meat": 1})
    labels = labels[~labels.index.duplicated(keep="first")]
    return labels
