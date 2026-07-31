# results/

## best

The study's 7 curated "best" node2vec+ embeddings (chosen by hyperparameter
sweep), each trained/evaluated once with `RandomizedSearchCV`. Files are
flat, prefixed by a short hex tag per embedding (e.g. `wcksnlsg_*`) plus
`baseline_logging.txt` for the raw-feature (no-embedding) baseline model —
see `scripts/train_baseline_models.py`. Each tag's `*_logging.txt` records
its node2vec p/q/gamma and the classifier's chosen hyperparameters and
cross-validated scores; `*_model.pkl` and `*_model_weights.txt` are the
fitted classifier. Produced 2024-12; the environment that generated these
predates this repo's `environment.yml` (first committed 2025-05), so exact
package versions aren't pinned/recorded for this run.

## deployment_for_permutations

The same 7 curated embeddings, retrained for compatbility with permutation
testing pipeline. One subdirectory per tag (e.g. `wcksnlsg/`).
`majority_hits_4.tsv`, `majority_hits_7.tsv`, and `num_hits.tsv` (top level) are 
`scripts/compute_majority_hits.py`'s per-feature vote counts across the 7 embeddings' 
predictions (plain-majority ≥4/7 and unanimous 7/7 thresholds).

Re-running the same 7 configs here versus `results/best` should reproduce
scores almost exactly, but is not bit-identical due to scikit-learn version differences
(1.3.2 vs the current 1.6.1). See
`notebooks/2026-07-31_best_vs_deployment_agreement.ipynb` for the
per-tag/per-task score, hyperparameter, and solver-convergence comparison
between the two directories.

## differential_abundance

Baseline per-feature differential abundance results (Mann-Whitney U test,
fold change, FDR) comparing meat vs. dairy diet groups and time points,
computed directly from raw feature abundances — independent of any
embedding/classifier. One file per feature type × label
(`{diet,time}_{metabolites,microbiome}.txt`). This is the baseline that
`scripts/da_comparison.py` / `scripts/significant_features_table.py` compare
the permutation-test consensus results against. Created outside of this repo.

## permutations

Permutation-testing pipeline output (`src/permutation.py`), which asks
whether the deployment models' feature-level consensus scores
(`results/deployment_for_permutations`) are more extreme than chance.
Per label (`diet/`, `time/`), running the full pipeline produces:

- `fit/` — `scripts/fit_permutation_test.py`'s one-time setup:
  `fitted_state.pkl` (the fitted `PermutationTest`) and `observed.tsv`
  (observed consensus scores).
- `100000_permutations/` — `scripts/run_permutations.py`'s 100k-permutation
  null:
  - `batches/batch_<NNNN>.npz` — one file per `run_permutation_batch.py`
    chunk (1000 permutations/batch by default), holding that chunk's null
    scores under every `CONSENSUS_MODES` mode. This `batches/` directory is
    gitignored — it's the raw, reproducible-from-scratch intermediate the
    `combined_<mode>.tsv` files below are built from.
  - `combined_<mode>.tsv` — one per consensus mode (`hit_fraction_z`,
    `mean_z`, `median_z`, `max_z`, `mean_prob`, `median_prob`,
    `mean_confidence`, `median_confidence`, `hit_fraction_prob`), written by
    `scripts/combine_permutations.py`: every feature's observed score,
    p-value, and q-value under that mode.
  - `manifest.json` — batch/seed bookkeeping (also used to `--extend` a run
    to more permutations without recomputing existing batches).

Top-level `{diet,time}_significant_mean_confidence.tsv` are
`scripts/significant_features_table.py`'s report of which features came out
significant in the 100k-permutation run, for the `mean_confidence` consensus
mode.
