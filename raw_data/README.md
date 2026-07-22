
This directory is the complete raw-data archive for the project - everything the
analysis was originally built from. For just the files needed to run the
deployment pipeline (curated embeddings, the edge list, feature-ID lists), see
[`data/README.md`](../data/README.md) instead.

- `sample_breakdown.csv` - 5 train/test splits that have balanced baseline vs endpoint 
    and meat vs dairy samples. Both the baseline and endpoint samples for the same patients
    are always in the same fold. 
- `microbiome_info_data.csv` - metadata used for joining microbial and metabolite features,
    and for creating diet labels. 
- `microbe_metabolites_filtered_rank_normalized.csv` - a subset of rank normalized and merged 
    microbiome and metabolite feature data used to make non-embedding diet and time classifiers
- `unfiltered_micro_metab.csv` - raw abundance of microbes and metbolites before any QC or filtering.
- `All_micro_metab_KEY.csv` - Table of metabolite and microbiome feature identifiers linked to letter-numeric labels used in all files
- `metabolite_data_for_differential_abundance.csv` - metabolite feature abundance data used for differential abundance analysis and network creation 
- `microbiome_data_for_differential_abundance.csv` - microbiome feature abundance data used for differential abundance analysis and network creation 
