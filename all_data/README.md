
This directory is the complete raw-data archive for the project - everything the
analysis was originally built from. For just the files needed to run the
deployment pipeline (curated embeddings, the edge list, feature-ID lists), see
[`data/README.md`](../data/README.md) instead.

1. `raw/`

    - `sample_breakdown.csv` - 5 train/test splits with balanced baseline vs endpoint samples
    - `microbiome_info_data.csv` - diet group and other metadata
    - `microbe_metabolites_filtered_rank_normalized.csv` - rank normalized and merged microbiome and metabolite feature data used to make non-embedding diet and time classifiers
    - `unfiltered_micro_metab.csv` - raw abundance of microbes and metbolites
    - `All_micro_metab_KEY.csv` - Table of metabolite and microbiome feature identifiers linked to letter-numeric labels used in all files
    - `metabolite_data_for_differential_abundance.csv` - metabolite feature abundance data used for differential abundance analysis
    - `microbiome_data_for_differential_abundance.csv` - microbiome feature abundance data used for differential abundance analysis
