
This directory holds the minimal set of files needed to run the deployment
pipeline (`scripts/train_deployment_models.py`) - not the full raw-data
archive, which lives in [`all_data/`](../all_data/README.md).

1. `emb/`

    - the 7 curated "best" node2vec+ embedding spaces (selected from the
      hyperparameter sweep) used to train deployment models and generate
      feature predictions
    - filenames encode the node2vec+ parameters used to generate them:
      `emb_p_{p}_q_{q}_g_{g}.tsv.gz`

2. `edges.tsv`

    - the sample-feature edge list used to generate node2vec+ embeddings
      (edge weight = normalized count between a sample and a feature)

3. `nodes/`

    - `samples.txt` - sample node IDs (one per line) - which rows of the
      joint sample+feature embedding are training samples
    - `microbes.txt` - microbial feature IDs (one per line)
    - `metabolites.txt` - metabolite feature IDs (one per line)
    - passed to `Dataset.from_label_tsv` as `samples_path`/`feature_paths`
      (samples.txt for the former, microbes.txt + metabolites.txt for the
      latter) to explicitly say which embedding rows are samples vs. which
      feature type, instead of relying on a naming convention (e.g. an "MD"
      prefix). `microbes.txt`/`metabolites.txt` are also used to split
      feature-level predictions by type before z-scoring
      (`FeatureZScorer`).
    - `Dataset.from_label_tsv` raises an error if a listed sample has no
      label or no row in the embedding, or if a listed feature has no row
      in the embedding, and warns if the embedding or labels contain rows
      not listed in any of these files

4. `node_splits.tsv`

    - two columns: `node`, `split`
    - `split` is the **outer** cross-validation fold (1-5) in which that
      node is held out as a genuine test set; `-1` marks nodes that are
      never held out (still valid training data for every fold, just never
      evaluated on)
    - these are hand-defined, fixed folds - see "Nested cross-validation"
      below - generated from `all_data/raw/sample_breakdown.csv` by
      `scripts/generate_splits.py`. 
    - shared across every label type, so fold assignment is stored once
      rather than duplicated into each label file

5. `time_labels.tsv`, `diet_labels.tsv`

    - two columns each: `node`, `label` (binary 0/1)
    - generated from `all_data/raw/sample_breakdown.csv` and
      `all_data/raw/microbiome_info_data.csv` by `scripts/sample_labels.py`
    - joined against `node_splits.tsv` at load time (see
      `src/dataset.py`'s `Dataset.from_label_tsv`) - a label tsv on its own
      doesn't carry fold information

## Nested cross-validation

The deployment/sweep pipelines use a **nested** CV design:

- **Outer folds are defined once ahead of time.** `node_splits.tsv`
  defines for which split a sample should be held-out and used for testing. 
  `scripts/generate_splits.py` builds this file directly from
  `sample_breakdown.csv`'s existing `run`/`partition` columns to enforce the expected 
  formatting. We designed our folds such that the baseline and endpoint 
  samples from the same patient are in the same fold, and each fold must have a 
  balanced ratio of meat/dairy endpoint samples. These folds should always be stratified 
  for class label (and optionally any additioanl properties - age, batch, site, etc.).
- **Inner CV is automatic.** Within a given outer fold's training data, the
  sweep pipeline (`scripts/model.py`, `Classifier.run_sweep`) runs its own
  hyperparameter search using a plain integer `cv_folds` (see
  `src/dataset.py`'s docstring) - since the classifier is a
  `LogisticRegression`, scikit-learn automatically uses label-stratified
  k-fold for this, with no extra configuration. Sratification for other attributes is
  not currently implented for inner folds.

**Selecting an embedding space.** Selection is based only on the inner CV validation 
performance: for each outer fold, inner CV on that fold's training data gives one median 
validation score per outer fold; the best embedding space(s) are chosen by looking at the
median validation perofmrance and IQR across all 5 outer folds. 

**Assessing generalizability to new samples.** For each embedding space that passed our 
validation performance and IQR thresholds, we use the best performing hyper-parameters to 
train one model per outer-fold and report the performance on the unseen test samples 
for that fold. 

**Using deployment models to prioritize features.** The deployment pipeline 
(`scripts/train_deployment_models.py`) is intended for feature prioritization not 
sample classification, and runs on an already-selected embedding space. 
It uses the outer folds (`PredefinedSplit`) as the validation splits for
a hyperparameter search, then trains the final deployment model on all available samples. 
