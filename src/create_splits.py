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


for fold in [1,2,3,4,5]:
    os.makedirs(f'../data/cv_folds/{fold}', exist_ok=True)
    all_splits = pd.read_csv('../data/from_adelle/sample_breakdown.csv')
    splits = all_splits[all_splits['run'] == fold]
    #splits['Time'] = splits['Time'].map({'Baseline': 0, 'Endpoint': 1})

    train_ids = splits[splits['partition'] == 'train']['nodes'].reset_index(drop=True)
    train_labels = splits[splits['partition'] == 'train']['Time']
    train_data = pd.DataFrame(0, index=train_ids, columns=['Column1', 'Column2', 'Column3'])

    skf = StratifiedKFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(skf.split(train_data, train_labels)):
        with open(f'../data/cv_folds/{fold}/train_{i}.txt', 'w') as f:
            for idx in train_index:
                f.write(f'{train_ids[idx]}\n')
        with open(f'../data/cv_folds/{fold}/test_{i}.txt', 'w') as f:
            for idx in test_index:
                f.write(f'{train_ids[idx]}\n')