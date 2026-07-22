"""
Author: Keenan Manpearl
Date: 2024-09-09

Z-scores prediction probabilities within named feature subsets (e.g. microbial
vs metabolite), given feature ID lists. Works on any prediction DataFrame
indexed by feature ID, not specific to any one Dataset or pipeline.

"""

import os

import pandas as pd


class FeatureZScorer:
    """
    Z-scores predictions separately within each named subset of features.
    """

    def __init__(self, feature_lists: dict[str, list[str]]) -> None:
        self.feature_lists = feature_lists

    @classmethod
    def from_files(cls, file_paths: dict[str, str]) -> "FeatureZScorer":
        """
        build a FeatureZScorer from files listing feature IDs, one per line,
        e.g. {"microbes": "data/nodes/microbes.txt", "metabolites": "data/nodes/metabolites.txt"}
        """
        feature_lists = {}
        for name, fp in file_paths.items():
            with open(fp) as f:
                feature_lists[name] = f.read().splitlines()
        return cls(feature_lists)

    def score(self, preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        for each named feature subset, z-score preds within that subset
        """
        scored = {}
        for name, features in self.feature_lists.items():
            subset = preds.loc[features]
            scored[name] = (subset - subset.mean()) / subset.std()
        return scored

    def score_and_save(
        self, preds: pd.DataFrame, out_root: str, tag: str, model_type: str
    ) -> None:
        """
        z-score preds within each named feature subset and save each to its own tsv
        """
        zscore_dir = f"{out_root}/zscores"
        os.makedirs(zscore_dir, exist_ok=True)
        for name, df in self.score(preds).items():
            df.to_csv(
                f"{zscore_dir}/{tag}_{model_type}_{name}_predictions_zscored.tsv",
                sep="\t",
            )
