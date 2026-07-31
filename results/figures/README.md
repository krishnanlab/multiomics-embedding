# results/figures

Figures and summaries comparing four ways of calling a feature "associated" with `diet` or `time`:

- **Differential Abundance (DA)** — a per-feature statistical test (paired
  Wilcoxon for `time`, Mann-Whitney U for `diet`), precomputed outside this
  repo. Significant at `FDR < 0.05`.
- **Permutation Testing** — `src/permutation.py`'s classifier-consensus
  permutation test across 7 curated node2vec+ embeddings. Significant at
  raw `p < 0.05`. Has 9 sub-variants ("modes": `hit_fraction_z`, `mean_z`,
  `median_z`, `max_z`, `mean_prob`, `median_prob`, `mean_confidence`,
  `median_confidence`, `hit_fraction_prob`) that combine the 7 embeddings'
  per-feature scores differently; `mean_confidence` is the one used in the
  confusion-matrix figures because it has the second highest overlap coeffecient
  but recovered ~2-3X as many features as the metric with the highest overlap.
- **Majority Vote** — a feature counts as associated with a direction if
  `>=4` of the 7 deployment embeddings' models gave it `probability > 0.5`
  for that direction.
- **Unanimous Vote** — same rule, `>=7/7` (all embeddings agree).

## Data sources

| Data | Path | Produced by |
|---|---|---|
| DA tables | `results/differential_abundance/{diet,time}_{microbiome,metabolites}.txt` | precomputed outside this repo |
| Permutation testing (all 9 modes) | `results/permutations/{diet,time}/100000_permutations/combined_<mode>.tsv` | `scripts/run_permutations.py` + `scripts/combine_permutations.py` (via `run/10_local_fit_permutation_test_{diet,time}.sh` and the batch-run scripts) |
| Per-embedding deployment predictions | `results/deployment_for_permutations/<tag>/{diet,time}_feature_predictions.tsv` — one `<tag>` subdirectory per curated embedding | `scripts/train_deployment_models.py` |
| Per-feature hit counts (majority/unanimous vote input) | `results/deployment_for_permutations/num_hits.tsv` (full counts), `majority_hits_4.tsv`/`majority_hits_7.tsv` (thresholded reports) | `scripts/compute_majority_hits.py` |
| Feature ID lists (per omics type) | `data/nodes/{microbes,metabolites}.txt` | `scripts/build_feature_graph.py` |
| Per-feature sample coverage (missingness input) | `raw_data/edges.tsv` (bipartite sample-feature edge list) | precomputed outside this repo |

`num_hits.tsv` is a wide table: index = feature ID, columns
`{diet,time}_{dairy,meat,baseline,endpoint}` = count of the 7 deployment
embeddings where that direction's probability was `> 0.5` (0-7).
`majority_hits_4.tsv`/`majority_hits_7.tsv` are convenience reports listing
features that clear the majority (>=4) or unanimous (>=7) threshold in *any*
direction — note `majority_hits_4.tsv` trivially contains every feature: with
7 embeddings and a binary class, one direction always gets >=4 votes by
pigeonhole, so ">=4 in any direction" is the whole population. It's still a
meaningful, non-trivial filter in the confusion-matrix/summary outputs below
because those always check one *specific* direction's hit count, not "any
direction".

Diet DA found **zero** significant features in either omics type, so DA comparisons 
are only generated for `time`. Permutation Testing / Majority Vote / Unanimous Vote 
are still compared against each other for `diet`.

## Confusion matrices (`cm_*.png`)

Produced by `scripts/figures/confusion_matrices.py` (run
`scripts/compute_majority_hits.py` first). One PNG per comparison per label,
each a 2x2 grid of confusion matrices (rows = direction, columns = omics
type — `microbiome`/`metabolites`). Cell counts are TP/FN/FP/TN between the
row method and the column method, colored on a log scale; axis tick labels
are "Associated"/"Not Associated" and the axis title names the method.

**The x-axis (columns) always follows a fixed preference order: Differential
Abundance, then Permutation Testing, then Majority Vote, then Unanimous
Vote.** Whichever of the two methods being compared comes later in that
order is the column/x-axis — e.g. DA vs. Permutation Testing puts DA on the
row (y) and Permutation Testing on the column (x); Majority Vote vs.
Unanimous Vote puts Majority Vote on the row and Unanimous Vote on the
column.

9 PNGs: `cm_da_vs_permutation_testing_time`, `cm_da_vs_majority_vote_time`,
`cm_da_vs_unanimous_vote_time`, `cm_permutation_testing_vs_majority_vote_{diet,time}`,
`cm_permutation_testing_vs_unanimous_vote_{diet,time}`,
`cm_majority_vote_vs_unanimous_vote_{diet,time}`.

Shared plot style (600 DPI, tight margins) is `scripts/figures/style.py`,
imported by every script in `scripts/figures/`.

## Segment covariates (`effect_size_da_vs_*.png`, `pct_missing_da_vs_*.png`)

Produced by `scripts/figures/segment_covariates.py`, generalizing the
segment-boxplot cells in `notebooks/2026-07-30_condensed_summary.ipynb` from
permutation-testing-only to all three methods. `time` only, for the same
reason the DA confusion matrices are `time`-only.

For each method (Permutation Testing, Majority Vote, Unanimous Vote), splits
every feature into three detection-outcome segments relative to DA —
differential abundance only, both methods, method only — and boxplots two
per-feature covariates within each segment (with pairwise Mann-Whitney U
significance brackets between every pair of segments):

- `effect_size_da_vs_<method>_time.png` — DA's `|Fold Change|` (log2). Are
  the features a method uniquely catches (or uniquely misses) systematically
  weaker/stronger DA effects than the features both methods agree on?
- `pct_missing_da_vs_<method>_time.png` — percent of samples missing a
  measurement for that feature (`100 * (1 - graph degree / total samples)`).
  Are uniquely-caught/uniquely-missed features systematically more or less
  measured than the ones both methods agree on? The microbiome/metabolites
  subplots share a y-axis, since missingness is being compared across omics
  types here (unlike effect size, which is on a different scale per omics
  type).

Each of these PNGs is one figure with 1x2 subplots (microbiome, metabolites)
sharing the segment color convention. 6 PNGs total (2 covariates x 3
methods): `effect_size_da_vs_permutation_testing_time`,
`effect_size_da_vs_majority_vote_time`,
`effect_size_da_vs_unanimous_vote_time`,
`pct_missing_da_vs_permutation_testing_time`,
`pct_missing_da_vs_majority_vote_time`,
`pct_missing_da_vs_unanimous_vote_time`.

`pct_missing_all_features.png` is a separate, unsegmented reference figure -
one panel, every feature of each omics type (not restricted to DA/method
detection outcome), so the segments above can be read against each omics
type's overall missingness rather than just against each other.

## Metric summaries (`summaries/<group>/*.{tsv,png}`)

Produced by `scripts/figures/summarize_permutation_metrics.py`, generalizing
the mode x category comparison in
`notebooks/2026-07-28_time_consensus_mode_comparison.ipynb` into a script.
`time` only, for the same reason the DA confusion matrices are `time`-only.

Two subdirectories, each with 7 metrics — `n_features_identified`, `jaccard`,
`overlap_coefficient`, `direction_agreement`, `accuracy`, `precision`,
`recall` — as both a TSV and a heatmap PNG of the same table
(`<metric>.tsv`/`<metric>.png`), every method/mode together in one sheet per
metric rather than split across per-method files:

- `permutation_modes/` — one row per `combined_<mode>.tsv` consensus mode (9
  rows: `hit_fraction_z`, `mean_z`, `median_z`, `max_z`, `mean_prob`,
  `median_prob`, `mean_confidence`, `median_confidence`, `hit_fraction_prob`).
- `consensus_modes/` — 3 rows, the three headline ways of reaching consensus
  across the 7 embeddings: `mean_confidence` (the permutation-testing mode
  used everywhere else in `results/figures`), `majority_vote`, and
  `unanimous_vote` — a direct, uncluttered comparison without the other 8
  permutation modes' noise.

Every table's columns are `<omics>-<direction>` (direction-specific:
`microbiome-5-month-old`, `microbiome-12-month-old`,
`metabolites-5-month-old`, `metabolites-12-month-old`), `<omics> (combined)`
(both directions pooled, per omics type), and `everything (combined)` (both
omics types and both directions pooled). `direction_agreement` is only
meaningful for `(combined)` columns (trivially ~1.0 for a direction-specific
column by construction) and is `NaN`/"n/a" elsewhere. `accuracy`/`precision`/
`recall` use DA in the "actual" role and the method in the "predicted" role
purely as a labeling convention (not a claim that DA is more correct), over
the full DA feature universe for that column (not just the significant
subset). See `summaries/README.md` for what each metric means.
Each heatmap's color limits are data-derived — that table's own min/max,
ignoring `NaN` — not each metric's theoretical 0-1 range, so the color scale
always uses its full contrast on whatever that particular table contains;
`n_features_identified` still gets a log scale (its values span single
digits to 25,900, even within one table).
Heatmaps use a fixed linear 0-1 color scale for every metric except
`n_features_identified` (log scale — its values span single digits to
25,900), both drawn from `scripts/figures/style.py`.
