"""
Author: Keenan Manpearl
Date: 2026-07-21

Study-specific wiring for SweepRunner/DeploymentRunner: builds the
time/diet Tasks and points at this study's node lists (data/{label}_labels.tsv,
data/node_splits.tsv, data/nodes/*.txt - see sample_labels.py/generate_splits.py
to regenerate). Neither runner knows what "time"/"diet" mean - just data
in the format each expects.

"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.deployment import DeploymentRunner, DeploymentTask
from src.sweep import Task, SweepRunner
from sample_labels import load_diet_labels, load_time_labels
from generate_splits import load_node_splits

DEFAULT_EDG_FILE = "data/edges.tsv"
DEFAULT_SAMPLES_PATH = "data/nodes/samples.txt"
DEFAULT_FEATURE_PATHS = ["data/nodes/microbes.txt", "data/nodes/metabolites.txt"]
DEFAULT_SPLIT_TSV = "data/node_splits.tsv"

PRED_COLUMNS = {"time": ["baseline", "endpoint"], "diet": ["dairy", "meat"]}


def build_sweep_runner(edg_file: str = DEFAULT_EDG_FILE, **kwargs) -> SweepRunner:
    """build a SweepRunner configured for this study's time/diet classifiers"""
    labels = {
        "time": Task(labels=load_time_labels(), pred_columns=PRED_COLUMNS["time"]),
        "diet": Task(labels=load_diet_labels(), pred_columns=PRED_COLUMNS["diet"]),
    }
    node_splits = load_node_splits()
    return SweepRunner(
        edg_file=edg_file, node_splits=node_splits, labels=labels, **kwargs
    )


def build_deployment_runner(
    edg_file: str = DEFAULT_EDG_FILE, **kwargs
) -> DeploymentRunner:
    """
    DeploymentRunner for this study's time/diet classifiers. time_labels.tsv
    covers every sample (Base+End), validated against samples.txt;
    diet_labels.tsv only covers endpoint samples, so its task has no
    samples_path (trusts its own rows instead).
    """
    labels = {
        "time": DeploymentTask(
            label_tsv="data/time_labels.tsv",
            pred_columns=PRED_COLUMNS["time"],
            samples_path=DEFAULT_SAMPLES_PATH,
        ),
        "diet": DeploymentTask(
            label_tsv="data/diet_labels.tsv", pred_columns=PRED_COLUMNS["diet"]
        ),
    }
    return DeploymentRunner(
        edg_file=edg_file,
        split_tsv=DEFAULT_SPLIT_TSV,
        feature_paths=DEFAULT_FEATURE_PATHS,
        labels=labels,
        **kwargs,
    )
