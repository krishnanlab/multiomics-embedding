'''
Author: Keenan Manpearl
Date: 2024-09-03

This script creates 5 cross validation folds per fold from adelle
for hyperparameter tuning within each fold. 
This is to keep folds consistent across all hyperparameter searches.
'''

import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold


NSPLITS = 10

for model in [1,2,3,4,5]:
    # make dir to save splits
    os.makedirs(f'../data/cv_folds/{model}', exist_ok=True)
    # read in sample breakdown file from addelle
    all_splits = pd.read_csv('../data/raw/sample_breakdown.csv')
    # get splits for specied model 
    splits = all_splits[all_splits['run'] == model]

    # get taining IDS
    train_ids = splits[splits['partition'] == 'train']['nodes'].reset_index(drop=True)
    # get time point for training IDS
    train_labels = splits[splits['partition'] == 'train']['Time']
    # fake training data for making plits
    train_data = pd.DataFrame(0, index=train_ids, columns=['Column1', 'Column2', 'Column3'])

    skf = StratifiedKFold(n_splits=NSPLITS)
    for i, (train_index, test_index) in enumerate(skf.split(train_data, train_labels)):
        with open(f'../data/cv_folds/{model}/train_{i}.txt', 'w') as f:
            for idx in train_index:
                f.write(f'{train_ids[idx]}\n')
        with open(f'../data/cv_folds/{model}/test_{i}.txt', 'w') as f:
            for idx in test_index:
                f.write(f'{train_ids[idx]}\n')