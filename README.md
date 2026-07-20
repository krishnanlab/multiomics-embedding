This repository contains code and data to accompany the paper **Network-based representation learning reveals the impact of age and diet on the gut microbial and metabolomic environment of U.S. infants in a randomized controlled feeding trial** [doi.org/10.1101/2024.11.01.621627](https://www.biorxiv.org/content/10.1101/2024.11.01.621627v1). This includes preprocessing the original microbial and metabolomic count data, creating a sample X feature edge list where the edge weight between two nodes is their normalized count value, creating node2vec+ embeddings, selecting embedding spaces, and using embeddings to train diet and time point classifiers. 


## Installation

All python package dependencies may be installed using conda. 
If you do not already have conda installed see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for installation instruction. 

Then run the following:

```
git clone git@github.com:krishnanlab/multiomics-embedding.git
cd multiomics-embedding
conda env create -f environment.yml
```


## Usage

For ease of use, `run/` contains shell scripts that call the python code in `scripts/`. 
All run scripts should be invoked from the project root.
Each script’s header comment documents its usage and any required arguments.

The pipeline runs in this order:

1. `run_initial_sweep.sh <wandb_username>` — sweep node2vec+ parameters (p, q, gamma) and evaluate their effect on time point classifiers only.
2. `run_joint_sweep.sh <wandb_username>` — sweep node2vec+ parameters and evaluate their effect on both time point and diet classifiers jointly.
3. `run_all.sh` — compare all unique embedding spaces generated during the two sweeps and select the top performers.
4. `run_baseline.sh` — train logistic regression models using the processed -omics counts directly as features, as a baseline for comparison against embedding-based models.
5. `run_deployment.sh` — train logistic regression models using the selected embedding features and identify -omics features predicted to be associated with a diet or time point phenotype.

## Data

`data/raw/` contains the microbiome and metabolomics abundance data underlying this project, plus sample metadata and CV splits; see [data/README.md](data/README.md) for a full description of each file. Briefly:

- Microbial features are gene functional annotations (KEGG `K#####` orthologs, `COG####` clusters, eggNOG `ENOG#...` groups) and taxonomic lineage strings (`k_...p_...c_...o_...f_...g_...s_...`).
- Metabolite features are anonymized compound IDs split by extraction method (`N_AQ.###` for aqueous, `N_LP.###` for lipid).

## Repository Organization 

```
├── data/            # raw and processed data
├── notebooks/       # exploratory analysis 
├── src/             # generic library: Dataset, LogisticRegressionClassifier, FeatureZScorer, embedding utilities
├── scripts/         # this study's pipelines (paths, column names, label definitions), built on src/
├── results/         # all results for top performing embedding spaces
├── run/             # shell scripts to call scripts/
├── environment.yml  # conda environment

```
In this repository we only include data and results for our top performing embedding spaces which were used in the paper. The performance of other embedding spaces can be seen in our [public wandb project](https://wandb.ai/keenan-manpearl/multiomics_embedding). Variation of all models is explored in [notebooks/2024-12-13_model_variance.ipynb](https://github.com/krishnanlab/multiomics-embedding/blob/main/notebooks/2024-12-13_model_variance.ipynb)

### `src/` vs `scripts/`

**`src/` is what you need if you want to use this method on your own multi-omics data.** It's a small, dataset-agnostic library with no knowledge of this study's file paths, column names, or label names - any number of feature types, any set of binary labels. See [Library Reference](#library-reference-src) below for the full API.

**`scripts/` is what reproduces our results.** It holds this study's specifics - reading `data/raw/sample_breakdown.csv`'s `partition`/`run`/`nodes` columns and `data/raw/microbiome_info_data.csv`'s `Group`/`Time` columns (`scripts/sample_labels.py`), this study's embedding file naming convention and the `MD`-prefix sample/feature split (`scripts/train_deployment_models.py`, `scripts/model.py`), and the two feature-type lists (`data/raw/microbes.txt`, `data/raw/metabolites.txt`) - and calls into `src/` to do the actual work. This is also where `run/`'s shell scripts point (`scripts/train_deployment_models.py`, `scripts/train_baseline_models.py`, `scripts/run_sweep.py`, `scripts/submit_all_embeddings.py`).

## Library Reference (`src/`)

This section documents every public class and method in `src/`, for anyone who wants to bring their own multi-omics dataset. Anything not listed here (e.g. `LogisticRegressionClassifier._cv_fold_scores`, `embedding._embed_network`) is a private implementation detail, not part of the supported API.

### `src/dataset.py` - `Dataset`

A container for one binary classification target: a train split (and optionally a held-out test split) with labels, plus an optional matrix of extra rows to generate predictions for once a classifier is trained (e.g. feature-nodes that share an embedding space with the samples).

- **`Dataset(label_name, train_data, train_labels, cv_folds, test_data=None, test_labels=None, feature_matrix=None)`** - direct constructor. `train_data`/`test_data`: DataFrame or ndarray, rows = samples. `train_labels`/`test_labels`: 1-D binary (0/1) array-like, aligned with the corresponding data's rows. `cv_folds`: an int for plain k-fold, or any scikit-learn CV splitter (e.g. `PredefinedSplit`) usable as `RandomizedSearchCV`'s `cv` argument. `feature_matrix`: a DataFrame indexed by feature ID, only if your data source jointly embeds samples and features - otherwise leave it `None`.
- **`Dataset.from_tables(label_name, feature_table, labels, cv_folds, feature_matrix=None)`** *(classmethod)* - the easiest way to build a `Dataset`: pass in an already-loaded feature table and a `labels` Series, and it aligns `feature_table.loc[labels.index]` for you. This is what every script in `scripts/` uses.
- **`.features_to_predict()`** → `DataFrame | None` - returns `feature_matrix`.
- **`.with_shuffled_labels(rng)`** → `Dataset` - returns a copy with `train_labels` randomly permuted. This is the intended extension point for permutation testing: retrain a classifier on the shuffled copy via `fit_full`, predict on features via `predict_features`, and z-score the result via `FeatureZScorer.score` to build a null distribution.

### `src/classifier.py` - `LogisticRegressionClassifier`

Runs hyperparameter search, trains a final model, evaluates it, and (if the `Dataset` supports it) predicts on its feature matrix. One instance handles one binary label - `label_name` and `pred_columns` are just tags/config, nothing is subclassed per label.

- **`LogisticRegressionClassifier(label_name, pred_columns, seed, cv_max_iter, n_iter_search, fit_max_iter=None, scoring="f1", refit=True, param_distributions=None)`** - `pred_columns` is a 2-element list naming the two classes in the order `predict_proba` returns them (i.e. `[negative_class_name, positive_class_name]`). `param_distributions` defaults to a reasonable C/penalty/solver/l1_ratio search grid; override it to change the search space.
- **`.cv_search(dataset)`** → `self` - hyperparameter search via `RandomizedSearchCV`, validating on `dataset.cv_folds`. Populates `self.search_`.
- **`.fit_full(dataset, params=None)`** → `self` - trains a final model on all of `dataset.train_data` (defaults to the best params found by `cv_search`). Populates `self.model_`. This is the second half of the permutation-testing extension point: call it directly with the *original* `best_params` on a `with_shuffled_labels`-permuted dataset, without re-running `cv_search` each time.
- **`.evaluate(dataset, split)`** → `dict` - accuracy/balanced_accuracy/precision/recall/f1 for `"train"` or `"test"` (the latter requires `dataset.test_data`/`test_labels`).
- **`.predict_features(dataset)`** → `DataFrame | None` - predictions for `dataset.features_to_predict()`, or `None` if unsupported. The other half of the permutation-testing extension point.
- **`.write_results(file, n_folds)`** - writes best CV hyperparameters and per-fold validation scores (median/IQR/variance) to an already-open file. Requires `scoring` to have been a list of metric names.
- **`.run_sweep(dataset, n_folds, score_metric="score")`** → `SweepResult(best_params, best_score, fold_scores, train_metrics, test_metrics)` - the full sweep-evaluation procedure: `cv_search` → `fit_full` → `evaluate` on train and (if present) test. Use this when you want to score a hyperparameter setting.
- **`.run_deployment(dataset)`** → `DeploymentResult(best_params, feature_predictions)` - the full deployment procedure: `cv_search` → `fit_full` → `predict_features`. Use this when you want a final trained model plus (optionally) feature-level predictions; call `write_results`/`save`/`save_weights` afterward to persist output.
- **`.save(path)`** / **`.save_weights(path)`** - pickle the trained model / write its coefficients to a text file.

### `src/zscoring.py` - `FeatureZScorer`

Z-scores predictions within any number of named feature subsets (this study uses two: microbes and metabolites).

- **`FeatureZScorer(feature_lists)`** - `feature_lists` is a `dict[str, list[str]]` mapping a subset name to the feature IDs in it.
- **`FeatureZScorer.from_files(file_paths)`** *(classmethod)* - builds one from files listing feature IDs, one per line, e.g. `{"microbes": "data/raw/microbes.txt", "metabolites": "data/raw/metabolites.txt"}`.
- **`.score(preds)`** → `dict[str, DataFrame]` - z-scores `preds` within each named subset, in memory. The extension point for permutation testing (build a null distribution without writing every permutation to disk).
- **`.score_and_save(preds, out_root, tag, model_type)`** - calls `.score()` and writes each subset to `{out_root}/zscores/{tag}_{model_type}_{name}_predictions_zscored.tsv`.

### `src/embedding.py`

- **`load_or_create_embedding(edg_file, emb_file, n2v_mode, p, q, gamma, seed)`** → `DataFrame` - loads a cached node2vec+ embedding from `emb_file` if it exists, otherwise generates one from the `edg_file` edge list and caches it. Not specific to this study - takes a generic edge list, no hardcoded paths.

## Reproducing deployment z-score predictions

To get z-scored feature predictions for the selected ("best") deployment models - the microbe/metabolite predictions used downstream in the analysis notebooks - run:

```
bash run/run_deployment.sh
```

which is equivalent to `python scripts/train_deployment_models.py` from the project root. This iterates the 7 selected embedding parameter sets hardcoded in that script's `__main__` block and, for each, requires:

- a cached embedding at `data/best_emb/emb_p_{p}_q_{g}.tsv.gz` (already provided in this repo for the 7 selected spaces),
- `data/raw/microbes.txt` and `data/raw/metabolites.txt` (the feature-ID lists used to split predictions by feature type).

For each embedding and each classifier target (`time`, `diet`), it writes to `results/best_<date>/` (or `--out <dir>`):

- `<tag>_<target>_feature_predictions.tsv` - raw predicted probabilities for every microbe/metabolite feature.
- `zscores/<tag>_<target>_{microbes,metabolites}_predictions_zscored.tsv` - those predictions z-scored within each feature type.
- `<tag>_<target>_model.pkl` / `<tag>_<target>_model_weights.txt` - the trained model and its coefficients.
- `<tag>_logging.txt` - the CV hyperparameters and validation scores for both classifiers.


## License 
This repository and all its contents are released under the [BSD 3-Clause License](https://opensource.org/license/BSD-3-Clause); See [LICENSE](https://github.com/krishnanlab/multiomics-embedding/blob/main/LICENSE)

## Authors 
Adelle Price, Sakaiza Rasolofomanana-Rajery, Keenan Manpearl, Charles E. Robertson, Nancy F. Krebs, Daniel N. Frank, Arjun Krishnan*, Audrey E. Hendricks*, Minghua Tang*  
*These authors contributed equally.

## Funding
NIH (NIDDK) 1K01DK111665-01, 1R01DK126710, the Beef Checkoff through the National Cattlemen’s Beef Association, and the National Pork Board.

## Citation
The paper associated with this codebank is:

> Price A, Rasolofomanana-Rajery S, Manpearl K, Robertson CE, Krebs NF, Frank DN, Krishnan A, Hendricks AE, Tang M. Network-based representation learning reveals the impact of age and diet on the gut microbial and metabolomic environment of U.S. infants in a randomized controlled feeding trial. *bioRxiv*. 2024. doi: [10.1101/2024.11.01.621627](https://www.biorxiv.org/content/10.1101/2024.11.01.621627v1)

```bibtex
@article{price2024network,
  title={Network-based representation learning reveals the impact of age and diet on the gut microbial and metabolomic environment of U.S. infants in a randomized controlled feeding trial},
  author={Price, Adelle and Rasolofomanana-Rajery, Sakaiza and Manpearl, Keenan and Robertson, Charles E and Krebs, Nancy F and Frank, Daniel N and Krishnan, Arjun and Hendricks, Audrey E and Tang, Minghua},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.11.01.621627}
}
```
