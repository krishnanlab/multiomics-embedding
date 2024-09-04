Two files came directly from Adelle:

1. `unfiltered_micro_metab.csv` - this contains raw abundance of microbes and metbolites
2. `sample_breakdown.csv` - this contains 5 train/test splits with balanced baseline vs endpoint samples

In order to optimzie hyperparameters we are doing nested CV,
so for each of the 5 original splits we need to create 5 cross validaton folds
using the training samples only. 

These folds were created once using `src/create_splits.py` and held constant. 
The sample IDs used for training/testing for each fold can be found in
the `data/cv_folds/` folders. 
Each folder (labeled 1-5) contains 5 training node lists
and 5 testing node lists. 

`data/edg/` contains edge list files that were created with `feature_preprocessing.py`.
Each file has three columns: sample node, microbe/metbolite node and edge weight.
For each of the original 5 folds, the entire training set was used to
calculate a tau score to select microbes and metabolites that are variable across samples. 

The tau score was calculated as follows:

1. x_i = feature / featue_max
2. sum(1 - x_i) / (num_features - 1)

Microbes and metbolites with a tau >= 0.8 were retained.
A normalized edge weight was calculated as follows:

1. For each microbe/metbolite rank samples from most abundant (1)
to least abundant (N)
2. edge weight = (N - rank + 1) / N

All edges are between a sample and a microbe or a metbolite.
There are no edges between microbes and metabolites or between samples.
All sample nodes were used in graph creation, but only variable 
microbes and metbolites were retained, thus each fold may have different nodes. 