# results/figures/summaries

TSV + heatmap summaries of how well each  mode  agrees with
baseline differential abundance (DA) for `time`. See `../README.md` for
what's in each subdirectory and how the category columns are defined.
`precision`/`recall`/`accuracy` borrow their names and formulas from the
standard classification-metric convention.

- **`n_features_identified`** — how many features this method/mode calls
  associated. 
- **`jaccard`** — of every feature *either* DA or the method calls
  associated, what fraction do *both* agree on. The strictest overlap
  measure here: it's diluted whenever one side's set is much bigger than the
  other's, since the denominator (the union) grows with the larger set even
  if the smaller set is fully contained in it.
- **`overlap_coefficient`** — of whichever side (DA or the method) called
  *fewer* features associated, what fraction of those are also called by the
  other side. Unlike Jaccard, this isn't diluted when one side's list is
  much longer than the other's — it asks "does the smaller, stricter call
  get absorbed into the larger one," which is the more useful question when
  comparing a conservative method against a permissive one.
- **`direction_agreement`** — among features *both* DA and the method call
  associated, how often do they agree on *which group* (e.g. `Meat` vs.
  `Dairy`, `5-month-old` vs. `12-month-old`) the feature is higher in.
  Answers "when they agree it's a hit, do they agree on the direction of the
  effect" — a separate question from whether they agree it's a hit at all.
- **`accuracy`** — how often the method's associated/not-associated call
  matches DA's, across *every* feature (not just the ones either side flags).
  Easy to read too optimistically: with tens of thousands of features that
  are mostly not associated. 
- **`precision`** — of the features *this method* calls associated, what
  fraction does DA also call associated. How likely is DA to agree on a feature
  identified through this method?
- **`recall`** — of the features *DA* calls associated, what fraction does
  this method also catch. How likely are we to pick up a feature that DA identified?
