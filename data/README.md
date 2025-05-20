
1. `best_emb/`
    
    - final embedding spaces used to make predictions
    - name represents node2vec+ parameters used to generate embedding space

2. `cv_folds/`

    In order to optimzie hyperparameters we are doing nested CV,
    so for each of the 5 original splits we need to create 10 cross validaton folds
    using the training samples only. 

    These folds were created once using `src/create_splits.py` and held constant. 
    The sample IDs used for training/testing for each fold can be found in
    the `data/cv_folds/` folders. 
    Each folder (labeled 1-5) contains training node lists
    and testing node lists. 

3. `edg`

    -`full_data_pecan_3.tsv` - edge list used to create embeddings

4. `raw/`

    - `microbe_metabolites_filtered_rank_normalized.csv` - used to make edge list
    - `unfiltered_micro_metab.csv` - raw abundance of microbes and metbolites
    - `microbiome_info_data.csv` - diet group and other metadata
    - `sample_breakdown.csv` - 5 train/test splits with balanced baseline vs endpoint samples

