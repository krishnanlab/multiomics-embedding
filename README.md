This repository contains code and data to accompany the paper **Network-based
representation learning reveals the impact of age and diet on the gut microbial
and metabolomic environment of U.S. infants in a randomized controlled feeding
trial** [doi.org/10.1101/2024.11.01.621627](https://www.biorxiv.org/content/10.1101/2024.11.01.621627v1).
This includes preprocessing the original microbial and metabolomic count data,
creating a sample X feature edge list where the edge weight between two nodes
is their normalized count value, creating node2vec+ embeddings, selecting
embedding spaces, and using embeddings to train diet and time point classifiers.

## Prerequisites

- **conda** - if you don't already have it, install it
  [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).
- **A wandb account, logged in** (`wandb login`) **(optional)** - only
  required for `scripts/run_sweep.py` (a wandb-driven random search, no
  non-wandb mode - use `scripts/run_sweep_local.py` instead if you don't
  have one). `scripts/sweep.py` and `scripts/deploy.py` also work without
  one: pass `--no-wandb` to just print/return results locally instead of
  logging them.
- **SLURM / `sbatch`** **(optional)** - only needed for the `run/slurm_*.sh`
  scripts (see [`run/` naming](#run-naming-local_nn_-vs-slurm_nn_) below,
  `scripts/slurm_utils.py`). Also requires your own `jobs/template.sh`
  (gitignored, cluster-specific - SBATCH partition/account/qos and
  module/conda activation for your own HPC setup).

## Dependencies

All python package dependencies are installed using conda:

```
git clone git@github.com:krishnanlab/multiomics-embedding.git
cd multiomics-embedding
conda env create -f environment.yml
```

This installs:

- `python` 3.10
- `numpy`, `pandas`, `scipy` - data handling
- `scikit-learn` - `LogisticRegression`/`RandomizedSearchCV`
- `pecanpy` - node2vec+ embedding generation
- `matplotlib` - plotting in `notebooks/`
- `ipykernel` - running `notebooks/` in Jupyter
- `wandb` **(optional)** - see [Prerequisites](#prerequisites) above.

## Usage

`scripts/` holds every step of the pipeline; `run/` has thin shell wrappers
around them for this study's fixed invocations, and should be invoked from
the project root. Each `run/` script's header comment documents its usage.

### `run/` naming: `local_NN_` vs `slurm_NN_`

`run/` scripts are numbered in pipeline order (`01`-`08`). Steps 4-8 each
have a `local_NN_*.sh` and a `slurm_NN_*.sh` variant:

- **`local_NN_*.sh`** runs the step directly in your current shell (or as
  local concurrent subprocesses, for the sweep/evaluation steps -
  `--max_jobs 4`).
- **`slurm_NN_*.sh`** submits the same step to SLURM instead. For steps 4-5
  (the sweeps), each *trial* becomes its own SLURM job (via `--slurm`,
  `jobs/template.sh`, `scripts/slurm_utils.py`) - real parallelism, one job
  per (p, q, gamma) combo. For steps 6-8, the underlying script has no
  per-trial `--slurm` support, so the *whole script* is submitted as one
  SLURM job (`scripts/submit_job.py`) - not per-trial parallel, just queued
  instead of run inline.

Steps 4-5 additionally come in `_wandb` and non-`_wandb` variants -
`scripts/run_sweep.py` requires `wandb login`; `scripts/run_sweep_local.py`
doesn't and writes results as local JSONs instead.

Both `--slurm` variants need a working `jobs/template.sh` (gitignored,
cluster-specific - see [Prerequisites](#prerequisites)) 

### Pipeline steps

- **`run/local_01_build_feature_graph.sh`** (`scripts/build_feature_graph.py`)
  - builds the sample-feature graph.
  **In:** `raw_data/metabolite_data_for_differential_abundance.csv`,
  `raw_data/microbiome_data_for_differential_abundance.csv`,
  `raw_data/microbiome_info_data.csv`.
  **Out:** `data/edges.tsv`,
  `data/nodes/{samples,microbes,metabolites}.txt`.
- **`run/local_02_generate_splits.sh`** (`scripts/generate_splits.py`) -
  builds the outer CV fold assignment.
  **In:** `raw_data/sample_breakdown.csv`.
  **Out:** `data/node_splits.tsv`.
- **`run/local_03_sample_labels.sh`** (`scripts/sample_labels.py`) - builds
  the binary classification labels.
  **In:** `raw_data/sample_breakdown.csv`, `raw_data/microbiome_info_data.csv`.
  **Out:** `data/time_labels.tsv`, `data/diet_labels.tsv`.
- **`scripts/sweep.py`** / **`scripts/deploy.py`** (run directly, no `run/`
  wrapper - see [`src/` vs `scripts/`](#src-vs-scripts) above) - one
  node2vec+ embedding + classifier pass for a single given set of embedding
  parameters; the building block every sweep/evaluation step below calls
  under the hood.
  **In:** `--edges-file`/`--samples-file`/`--feature-files`/`--p`/`--q`/`--g`
  (or `--embedding-file` to skip generation).
  **Out:** a results dict (printed, or logged to wandb), plus - if
  `--save-to`/`--out` is given - the model, its weights, and a results JSON
  (`deploy.py` also writes feature-level predictions).
- **`run/{local,slurm}_04_initial_sweep{,_wandb}.sh`**
  (`scripts/run_sweep.py`/`scripts/run_sweep_local.py`, 100 trials, ranked
  by time-point validation F1) - random-searches node2vec+ parameters (p,
  q, gamma), running `scripts/sweep.py` once per trial.
  **Out:** every trial's embedding, cached to `emb_cache/`; results logged
  to wandb (`_wandb` variants) or written as local JSONs under `--out`.
- **`run/{local,slurm}_05_joint_sweep{,_wandb}.sh`** - same as step 4, but
  200 trials ranked by the combined time+diet score.
- **`run/{local,slurm}_06_evaluate_embeddings.sh`**
  (`scripts/submit_all_embeddings.py`) - re-evaluates every unique
  embedding steps 4-5 cached, so they can be compared and the top
  performers selected.
  **Out:** one results JSON per embedding under `--out`.
- **`run/{local,slurm}_07_train_baseline.sh`**
  (`scripts/train_baseline_models.py`) - trains logistic regression models
  directly on the raw -omics data (no embedding), as a baseline for
  comparison.
  **In:** `raw_data/microbe_metabolites_filtered_rank_normalized.csv`,
  `data/{time,diet}_labels.tsv`, `data/node_splits.tsv`,
  `data/nodes/samples.txt`.
  **Out:** `results/best/baseline_logging.txt`.
- **`run/{local,slurm}_08_train_deployment.sh`**
  (`scripts/train_deployment_models.py`) - fits a final deployment model
  per classifier for each of the 7 curated "best" embedding spaces, and
  z-scores the resulting feature predictions.
  **In:** the 7 curated embeddings (`data/emb/*.tsv.gz`),
  `data/nodes/{samples,microbes,metabolites}.txt`,
  `data/{time,diet}_labels.tsv`, `data/node_splits.tsv`.
  **Out:** `results/best_<date>/` - see
  [Reproducing deployment z-score predictions](#reproducing-deployment-z-score-predictions)
  below for the full file list.
- **`run/local_09_train_deployment_for_permutations.sh`** through
  **`run/local_12_combine_permutations_{time,diet}.sh`** - the
  significance-testing pipeline (per-feature label-permutation p-values/
  q-values for the 7 curated embeddings) - see
  [Significance testing](#significance-testing) below.

### Dataset size and resource requirements

The full graph (`data/edges.tsv`) has **26,009 nodes** - 109 samples (infant
timepoints) and 25,900 features (17,033 microbial, 8,867 metabolite) - and
**~2.39M edges** (~32 MB). The 7 curated "best" embeddings
(`data/emb/*.tsv.gz`, 128-dim each) total ~145 MB.

Time and peak memory (`MaxRSS`, measured via `sacct`/`sstat`, not just what
was requested) per step - all well under `scripts/slurm_utils.py`'s 5 GB
default, so only raise `--slurm-mem` for a larger `dim`:

| Step | Time | Peak memory |
|---|---|---|
| One embedding, cold cache (steps 4-5, generation only) | ~90 min @ 4 CPUs / ~48 min @ 8 | ~1 GB |
| One `sweep.py` trial, cold cache (steps 4-5, incl. nested-CV search) | ~1.5 hours | ~1 GB |
| One embedding re-score, cached (step 6, `--embedding-file`) | ~2-3 min | ~1 GB |
| `train_deployment_models.py`, all 7 embeddings (step 8) | ~3 min total | ~1.25 GB |
| One permutation trial (measured via `sacct` across a real 1000-batch run - see the `slurm-job-sizing` skill) | ~0.7-0.8s | ~1 GB fixed + ~1.4 MB/trial in the batch buffer |
| One permutation-batch worker (loads a pickled `PermutationTest`), default `--batch-size 1000` | ~13-15 min | ~2.5 GB |

Embedding-generation figures assume pecanpy's default `dim`/`num_walks`/
`walk_length`/`window_size`; both time and memory scale with these - a
larger `dim` or `walk_length` will cost more of both.

Step 4 (100 trials) / step 5 (200 trials) locally at `--max_jobs 4`: roughly
`(trials / 4) x 1.5 hours` - **~37 hours** / **~75 hours**. Slurm recommended. 

## Data

There are two data directories:

- **`raw_data/`** is the complete raw-data archive - microbiome and
  metabolomics abundance data and sample metadata; see
  [raw_data/README.md](raw_data/README.md) for a full description of each file.
- **`data/`** is the minimal set of files actually needed to run the
  deployment pipeline - the 7 curated "best" node2vec+ embeddings, the edge
  list, and feature-ID lists; see [data/README.md](data/README.md).
  `scripts/build_feature_graph.py` builds `data/edges.tsv` and the feature-ID
  lists directly from `raw_data/`.

The feature types we use are:

- Microbial features are gene functional or phylogenetic annotations (KEGG
  `K#####` orthologs, `COG####` clusters, eggNOG `ENOG#...` groups) and
  taxonomic lineage strings (`k_...p_...c_...o_...f_...g_...s_...`).
- Metabolite features are anonymized compound IDs split by extraction method
  (`N_AQ.###` or `P_AQ.###` for aqueous, `N_LP.###` or `P_LP.###` for lipid).

## Repository Organization

```
├── raw_data/        # complete raw-data archive
├── data/            # minimal data needed to run the deployment pipeline
├── notebooks/       # exploratory analysis and figures
├── src/             # generic library: Dataset, LogisticRegressionClassifier, FeatureZScorer, embedding utilities
├── scripts/         # this study's pipelines (paths, column names, label definitions), built on src/
├── results/         # all results for top performing embedding spaces
├── run/             # shell scripts to call scripts/
├── environment.yml  # conda environment
```

In this repository we only include data and results for our top performing
embedding spaces which were used in the paper. The performance of other
embedding spaces can be seen in our
[public wandb project](https://wandb.ai/keenan-manpearl/multiomics_embedding).
Variation of all models is explored in
[notebooks/2024-12-13_model_variance.ipynb](https://github.com/krishnanlab/multiomics-embedding/blob/main/notebooks/2024-12-13_model_variance.ipynb).

### `src/` vs `scripts/`

**`src/` is what you need if you want to use this method on your own
multi-omics data.** It's a small, dataset-agnostic library that can be used to
predict -omic(s) features associated with one or more sample-level attributes.
See [Library Reference](#library-reference-src) below for the full API.

**`scripts/` is what reproduces our results.** It holds this study's
specifics - reading `raw_data/sample_breakdown.csv`'s `partition`/`run`/`nodes`
columns and `raw_data/microbiome_info_data.csv`'s `Group`/`Time` columns to
build `data/node_splits.tsv` (`scripts/generate_splits.py`) and
`data/{time,diet}_labels.tsv` (`scripts/sample_labels.py`), building
`data/edges.tsv` and the feature-ID lists from the raw abundance tables
(`scripts/build_feature_graph.py`), this study's embedding file naming
convention (`scripts/train_deployment_models.py`), and the explicit
sample/feature-ID lists (`data/nodes/samples.txt`, `data/nodes/microbes.txt`,
`data/nodes/metabolites.txt`) - and calls into `src/` to do the actual work.
This is also where `run/`'s shell scripts point
(`scripts/train_deployment_models.py`, `scripts/train_baseline_models.py`,
`scripts/run_sweep.py`/`scripts/run_sweep_local.py`,
`scripts/submit_all_embeddings.py`, plus `scripts/submit_job.py` for the
`run/slurm_*.sh` variants of steps with no per-trial `--slurm` of their own).

Two more scripts run a single node2vec+ embedding + classifier pass for one
given set of embedding parameters, independent of any particular sweep or
study wiring beyond `scripts/sweep_setup.py`'s label/split definitions:
`scripts/sweep.py` (nested-CV evaluation of one embedding space - the
"tuning" mode) and `scripts/deploy.py` (fits a final model on all data for
one embedding space - the "deployment" mode, no held-out test set). Both
require `--edges-file`, `--samples-file`, and `--feature-files` (validated
against the embedding/graph via `src/validation.py`) plus `--p`/`--q`/`--g`;
run either with `--help` for the full flag list.

## Library Reference (`src/`)

This section documents public classes and methods in `src/`, for anyone who
wants to bring their own multi-omics dataset.

### `src/dataset.py` - `Dataset`

A container for one binary classification target: a train split (and
optionally a held-out test split) with labels, plus an optional matrix of
extra rows to generate predictions for once a classifier is trained (e.g.
feature-nodes that share an embedding space with the samples).

- **`Dataset(label_name, train_data, train_labels, cv_folds, test_data=None,
  test_labels=None, feature_matrix=None)`** - direct constructor.
  `train_data`/`test_data`: DataFrame or ndarray, rows = samples.
  `train_labels`/`test_labels`: 1-D binary (0/1) array-like, aligned with the
  corresponding data's rows. `feature_matrix`: a DataFrame indexed by feature
  ID, to generate feature predictions with deployment models - otherwise
  leave it `None`.
- **`Dataset.from_tables(label_name, feature_table, labels, cv_folds,
  feature_matrix=None)`** *(classmethod)* - build a `Dataset` from an
  already-loaded feature table and a `labels` Series; aligns
  `feature_table.loc[labels.index]` for you.
- **`Dataset.from_label_tsv(label_name, feature_table, label_tsv, split_tsv,
  samples_path=None, feature_paths=None, no_split_sentinel=-1)`**
  *(classmethod)* - build a `Dataset` from tab-separated files instead of
  in-memory tables: `label_tsv` (columns `node`, `label`) joined against
  `split_tsv` (columns `node`, `split`). This is what every script in
  `scripts/` uses.
  **`cv_folds` and nested CV**: `split` in `split_tsv` defines the **outer**
  CV folds - which fold each node is held out in as a genuine test set, 
  stratified by class label(s) and any other property (see `data/README.md`);
  **`samples_path`/`feature_paths`**: plain-text files (one ID per
  line) telling `feature_table` which rows are training samples vs. extra
  feature-prediction rows. Every listed sample must have both a label and a row in
  `feature_table`; every listed feature must have a row in `feature_table` -
  otherwise this raises `ValueError`. Rows in `feature_table`/`label_tsv` not
  covered by either path raise a `UserWarning`.
- **`.features_to_predict()`** → `DataFrame | None` - returns
  `feature_matrix`.
- **`.with_shuffled_labels(rng)`** → `Dataset` - returns a copy with
  `train_labels` randomly permuted. This is the intended extension point for
  permutation testing: retrain a classifier on the shuffled copy via
  `fit_full`, predict on features via `predict_features`, and z-score the
  result via `FeatureZScorer.score` to build a null distribution.

### `src/classifier.py` - `LogisticRegressionClassifier`

Runs hyperparameter search, trains a final model, evaluates it, and (if the
`Dataset` supports it) predicts on its feature matrix. One instance handles
one binary label - `label_name` and `pred_columns` are just tags/config,
nothing is subclassed per label.

- **`LogisticRegressionClassifier(label_name, pred_columns, seed,
  cv_max_iter, n_iter_search, fit_max_iter=None, scoring="f1", refit=True,
  param_distributions=None)`** - `pred_columns` is a 2-element list naming
  the two classes in the order `predict_proba` returns them (i.e.
  `[negative_class_name, positive_class_name]`). `param_distributions`
  defaults to a reasonable C/penalty/solver/l1_ratio search grid; override it
  to change the search space.
- **`.cv_search(dataset)`** → `self` - hyperparameter search via
  `RandomizedSearchCV`, validating on `dataset.cv_folds`. Populates
  `self.search_`.
- **`.fit_full(dataset, params=None)`** → `self` - trains a final model on
  all of `dataset.train_data` (defaults to the best params found by
  `cv_search`). Populates `self.model_`. This is the second half of the
  permutation-testing extension point: call it directly with the *original*
  `best_params` on a `with_shuffled_labels`-permuted dataset, without
  re-running `cv_search` each time.
- **`.evaluate(dataset, split)`** → `dict` - accuracy/balanced_accuracy/
  precision/recall/f1 for `"train"` or `"test"` (the latter requires
  `dataset.test_data`/`test_labels`).
- **`.predict_features(dataset)`** → `DataFrame | None` - predictions for
  `dataset.features_to_predict()`, or `None` if unsupported. The other half
  of the permutation-testing extension point.
- **`.write_results(file, n_folds)`** - writes best CV hyperparameters and
  per-fold validation scores (median/IQR/variance) to an already-open file.
  Requires `scoring` to have been a list of metric name(s).
- **`.run_sweep(dataset, n_folds, score_metric="score")`** →
  `SweepResult(best_params, best_score, fold_scores, train_metrics,
  test_metrics)` - the full sweep-evaluation procedure: `cv_search` →
  `fit_full` → `evaluate` on train and (if present) test. Use this when you
  want to score a hyperparameter setting.
- **`.run_deployment(dataset)`** → `DeploymentResult(best_params,
  feature_predictions)` - the full deployment procedure: `cv_search` →
  `fit_full` → `predict_features`. Use this when you want a final trained
  model plus (optionally) feature-level predictions; call
  `write_results`/`save`/`save_weights` afterward to persist output.
- **`.save(path)`** / **`.save_weights(path)`** - pickle the trained model /
  write its coefficients to a text file.

### `src/zscoring.py` - `FeatureZScorer`

Z-scores predictions within any number of named feature subsets (this study
uses two: microbial and metabolite).

- **`FeatureZScorer(feature_lists)`** - `feature_lists` is a
  `dict[str, list[str]]` mapping a subset name to the feature IDs in it.
- **`FeatureZScorer.from_files(file_paths)`** *(classmethod)* - builds one
  from files listing feature IDs, one per line, e.g.
  `{"microbes": "data/nodes/microbes.txt", "metabolites":
  "data/nodes/metabolites.txt"}`.
- **`.score(preds)`** → `dict[str, DataFrame]` - z-scores `preds` within each
  named subset, in memory. The extension point for permutation testing
  (build a null distribution without writing every permutation to disk).
- **`.score_and_save(preds, out_root, tag, model_type)`** - calls `.score()`
  and writes each subset to
  `{out_root}/zscores/{tag}_{model_type}_{name}_predictions_zscored.tsv`.

### `src/embedding.py`

- **`load_or_create_embedding(edg_file, emb_file, n2v_mode, p, q, gamma,
  seed, dim=128, walk_length=80, window_size=10, workers=4)`** → `DataFrame`
  - loads a cached node2vec+ embedding from `emb_file` if it exists,
  otherwise generates one from the `edg_file` edge list and caches it. Not
  specific to this study - takes a generic edge list, no hardcoded paths.

### `src/sweep.py`

Generic node2vec+ -> classifier hyperparameter-sweep runner, not specific to
any one study.

- **`EmbeddingParams(p, q, gamma, dim=128, walk_length=80, window_size=10,
  n2v_mode="OTF", seed=42, workers=4)`** - identifies one embedding space;
  `.cache_tag(edg_file)` gives a filename-safe tag used for embedding caching
  and results file naming.
- **`Task(labels, pred_columns)`** - one binary classifier to train as part
  of a sweep run: `labels` is a `DataFrame` indexed by node with a single
  `label` column, `pred_columns` as in `LogisticRegressionClassifier`.
- **`BaseRunner`** - shared plumbing for `SweepRunner` and
  `src/deployment.py`'s `DeploymentRunner`: embedding generation/caching,
  classifier construction, results-JSON saving, wandb summary logging.
- **`SweepRunner(edg_file, node_splits, labels, num_outer_folds=None,
  inner_cv_folds=10, **kwargs)`** - `.run(params, log_wandb=False,
  save_models_to=None, embedding=None)` → `dict` - runs a nested-CV
  sweep-evaluation pass (one `LogisticRegressionClassifier.run_sweep` per
  outer fold per label) and returns median/IQR/mean validation F1 per label
  plus a `combined_score`.

### `src/deployment.py`

- **`DeploymentTask(label_tsv, pred_columns)`** - one binary classifier to
  train as part of a deployment run.
- **`DeploymentRunner(edg_file, split_tsv, samples_path, feature_paths,
  labels, scoring="f1", refit=True, **kwargs)`** *(extends `BaseRunner`)* -
  `.run(params, save_to=None, embedding=None, log_wandb=False)` → `dict` -
  fits a final classifier per label on all available data
  (`LogisticRegressionClassifier.run_deployment`), optionally saving the
  model/weights/feature-predictions/results JSON to `save_to`.

### `src/validation.py`

- **`read_graph_nodes(edg_file)`** → `set[str]` - unique node IDs referenced
  by an edge list.
- **`validate_node_lists(edg_file, list_files, embedding_nodes=None)`** - for
  every node named in a list file (e.g. samples/feature-ID lists), raises
  `ValueError` if it's missing from the graph (or from `embedding_nodes`, if
  given); warns (doesn't raise) about nodes present in the graph/embedding
  but not listed anywhere, since those are simply unused.

## Reproducing deployment z-score predictions

To get z-scored feature predictions for the selected ("best") deployment
models - the microbe/metabolite predictions used downstream in the analysis
notebooks - run:

```
bash run/local_08_train_deployment.sh
```

which is equivalent to `python scripts/train_deployment_models.py` from the
project root. This iterates the 7 selected embedding parameter sets
hardcoded in that script's `__main__` block and, for each, requires:

- a cached embedding at `data/emb/emb_p_{p}_q_{q}_g_{g}.tsv.gz` (already
  provided in this repo for the 7 selected spaces),
- `data/nodes/samples.txt`, `data/nodes/microbes.txt`, and
  `data/nodes/metabolites.txt` (explicit ID lists saying which embedding rows
  are samples vs. which feature type),
- `data/{time,diet}_labels.tsv` and `data/node_splits.tsv` (labels and CV
  fold assignment - see `data/README.md`; regenerate with
  `scripts/sample_labels.py` and `scripts/generate_splits.py` if needed).

For each embedding and each classifier target (`time`, `diet`), it writes to
`results/best_<date>/` (or `--out <dir>`):

- `<tag>_<target>_feature_predictions.tsv` - raw predicted probabilities for
  every microbe/metabolite feature.
- `zscores/<tag>_<target>_{microbes,metabolites}_predictions_zscored.tsv` -
  those predictions z-scored within each feature type.
- `<tag>_<target>_model.pkl` / `<tag>_<target>_model_weights.txt` - the
  trained model and its coefficients.
- `<tag>_logging.txt` - the CV hyperparameters and validation scores for both
  classifiers.

## Significance testing

Per-feature empirical p-values/q-values via label permutation (see
`src/permutation.py`), across the 7 curated deployment embeddings, for each
of `time`/`diet`. See `notebooks/2026-07-24_permutation_power_analysis.ipynb`
for how permutation count and assumed number of true positives trade off
against statistical power for each feature type (microbes vs. metabolites) -
worth reading before picking a permutation count.

Every permutation trial computes ALL of `src/permutation.py`'s
`CONSENSUS_MODES` at once - `hit_fraction` (fraction of embeddings with
`|z| >= threshold`), `mean_z`/`median_z`/`max_z`, and the raw-probability
modes `mean_prob`/`median_prob`/`n_confident` (count with
`predict_proba(reference_class) > prob_threshold`) - since the expensive
part (fitting the classifier on shuffled labels) is identical regardless
of which mode(s) you actually care about. Each batch's null scores are
saved as one `.npz` per mode, so `scripts/combine_permutations.py --mode
<name>...` can compute p-values/q-values for any subset of modes straight
from already-computed batches - no permutation ever needs to be rerun just
to look at a different summary statistic. `combined.tsv` is long-format,
one row per `(feature, mode)`.

Four steps, `run/local_09_*.sh` through `run/local_12_*.sh`:

1. **`run/local_09_train_deployment_for_permutations.sh`** - fits the 7
   deployment models with correctly-separated per-label log files
   (`results/deployment_for_permutations/`). Don't point the next step at
   `results/best/` - those files combine both labels in one
   `<tag>_logging.txt`, and `_parse_best_params` has no concept of
   sections, so it would silently mix time's and diet's hyperparameters
   together.
2. **`run/local_10_fit_permutation_test_{time,diet}.sh`** - one-time setup
   per label (the observed, unpermuted scores for every mode don't depend
   on permutation count, so this isn't repeated per tier).
3. **`run/{local,slurm}_11_run_permutations_{time,diet}.sh`** - runs
   10,000 permutations (10 batches of 1000, the `--batch-size 1000`
   default). The `slurm` variant submits each batch as its own job
   (~2 CPU-hours/label, ~4 total across both labels, at the real
   measured ~0.73s/trial - see the `slurm-job-sizing` skill; ~2.5GB
   peak RSS per batch, single-threaded).
4. **`run/local_12_combine_permutations_{time,diet}.sh`** - derives a
   1,000-permutation tier as the first 10 batches of the 10,000-permutation
   run (`scripts/slice_permutation_manifest.py` - no recomputation), then
   writes final p-value/q-value tables for both tiers:
   `results/permutations_{time,diet}_{1000,10000}/combined.tsv`.

All `results/permutations_*/` and `results/deployment_for_permutations/`
output is gitignored - it's fully regenerable from the `run/` scripts above.

At both tiers and for both labels, under the original `hit_fraction` mode
no feature cleared BH at q=0.05 (`min q-value = 1` everywhere) -
consistent with the power-analysis notebook, which shows 10,000
permutations only reaches power >= 0.8 with several dozen-to-hundred true
positives, not one or two. Those runs predate the multi-mode `.npz` batch
format above, so the `results/permutations_*` directories need
regenerating (steps 2-4) before comparing `hit_fraction` against the
other modes.

**Extending to 100,000 permutations later** doesn't require rerunning the
10,000 already computed: `scripts/run_permutations.py`'s `--extend` reuses
a previous manifest's batches unchanged (numpy's `SeedSequence` is
prefix-consistent given the same `--base-seed` and a fixed per-batch size,
verified this session) and only computes the new ones - keep `--batch-size`
the same (1000) across the extension so the existing batches stay aligned -
```
python scripts/run_permutations.py --fitted-state results/permutations_time_fit/fitted_state.pkl \
    --n-permutations 100000 --batch-size 1000 --base-seed 0 \
    --extend results/permutations_time_10000/manifest.json \
    --out results/permutations_time_100000 \
    --slurm --slurm-time 00:20:00 --slurm-mem 4GB --slurm-cpus 2
```
At the real measured rate (~0.73s/trial - see the `slurm-job-sizing`
skill; the estimate this section originally used was ~40x too slow),
100,000 permutations costs ~20 CPU-hours per label, ~90 net-new batch
jobs beyond the 10,000 tier's 10 (`--extend`, as above) - or well under
an hour wall-clock as a fresh, independent 100-batch submission if no
compatible 10,000-tier manifest exists yet to extend from. Both labels'
`results/permutations_{time,diet}_100000/` tiers in this repo were
produced as fresh submissions, not extended from the (now-superseded,
pre-multi-mode) 10,000 tier.

## Literature association lookup (LLM)

`scripts/feature_annotation.py` + `scripts/llm_feature_association.py` ask
Claude (with web search) whether a given microbe/metabolite feature has
published literature support for an association with a study group -
independent of this study's own statistics, which are only used upstream to
pick which features are worth asking about. See each script's module
docstring for the filtering rules and confidence rubric.

**Requires** `ANTHROPIC_API_KEY` set in the environment, and the `anthropic`
package (`environment.yml`).

Single lookup:

```
export ANTHROPIC_API_KEY=your-key-here
python scripts/llm_feature_association.py \
    --feature-id "k_Bacteria.p_Actinobacteria.c_Actinobacteria.o_Bifidobacteriales.f_Bifidobacteriaceae.g_Bifidobacterium.s_Unclassified.Bifidobacterium" \
    --omics-type microbiome --group infant_5mo \
    --out results/llm_annotations/smoke_test.json
```

(swap in an actual taxonomy ID from `raw_data/microbiome.txt` if that exact
one isn't present - `Bifidobacterium` at 5mo is a well-documented
association, good for eyeballing whether the JSON, confidence score, and
citations look sane.)

Batch (one feature ID per line in `--feature-list`):

```
python scripts/llm_feature_association.py \
    --feature-list results/permutations_diet_10000/meat_hits.txt \
    --omics-type metabolite --group infant_12mo_meat \
    --out-dir results/llm_annotations/
```

`--group` is one of `infant_5mo`, `infant_12mo`, `infant_12mo_meat`,
`infant_12mo_dairy` (see `GROUPS` in `scripts/llm_feature_association.py`
for the exact hypothesis text sent to the model).

## License

This repository and all its contents are released under the
[BSD 3-Clause License](https://opensource.org/license/BSD-3-Clause); see
[LICENSE](https://github.com/krishnanlab/multiomics-embedding/blob/main/LICENSE).

## Authors

Adelle Price, Sakaiza Rasolofomanana-Rajery, Keenan Manpearl, Charles E.
Robertson, Nancy F. Krebs, Daniel N. Frank, Arjun Krishnan*, Audrey E.
Hendricks*, Minghua Tang*  
*These authors contributed equally.

## Funding

NIH (NIDDK) 1K01DK111665-01, 1R01DK126710, the Beef Checkoff through the
National Cattlemen’s Beef Association, and the National Pork Board.

## Citation

The paper associated with this codebank is:

> Price A, Rasolofomanana-Rajery S, Manpearl K, Robertson CE, Krebs NF, Frank
> DN, Krishnan A, Hendricks AE, Tang M. Network-based representation learning
> reveals the impact of age and diet on the gut microbial and metabolomic
> environment of U.S. infants in a randomized controlled feeding trial.
> *bioRxiv*. 2024. doi:
> [10.1101/2024.11.01.621627](https://www.biorxiv.org/content/10.1101/2024.11.01.621627v1)

```bibtex
@article{price2024network,
  title={Network-based representation learning reveals the impact of age and diet on the gut microbial and metabolomic environment of U.S. infants in a randomized controlled feeding trial},
  author={Price, Adelle and Rasolofomanana-Rajery, Sakaiza and Manpearl, Keenan and Robertson, Charles E and Krebs, Nancy F and Frank, Daniel N and Krishnan, Arjun and Hendricks, Audrey E and Tang, Minghua},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.11.01.621627}
}
```
