'''
Author: Keenan Manpearl
Date: 2024-08-29

This script performs feature selection and normalization.
Then creates an edge list. 
'''


import pandas as pd
import matplotlib.pyplot as plt


# threshold for feature selection
# higher tau score indicates a feature is specfific to a subset of samples
# a lower score indicates a more uniform distribution of feature across all samples
TAU_THRESHOLD = 0.8

def calculate_tau(feature):
    x_i = feature / feature.max()
    n = len(feature)
    return sum(1 - x_i) / (n - 1)

# microbime and metabolite data
df = pd.read_csv('../data/from_adelle/unfiltered_micro_metab.csv')
df.index = df['sample.ID']
df = df.drop(columns=['Unnamed: 0', 'sample.ID'])

# train/test splits for each fold 
splits = pd.read_csv('../data/from_adelle/sample_breakdown.csv')

for fold in splits['run'].unique():
    # calculae tau scores considering only samples in training set for each fold
    samples = splits[(splits['run'] == fold) & (splits['partition'] == 'train')]['nodes']
    df_subset = df.loc[samples]
    tau_scores = df_subset.apply(calculate_tau)
    tau_scores.to_csv(f'../results/tau_scores/tau_scores_fold_{fold}.csv')
    tau_scores.hist()
    plt.xlabel('Tau Scores')
    plt.ylabel('Frequency')
    plt.title(f'Fold {fold}')
    plt.savefig(f'../results/tau_scores/tau_scores_histogram_fold_{fold}.png')
    plt.close()

    # feature selection and normalization 
    features = tau_scores[tau_scores > TAU_THRESHOLD].index.tolist()
    df_top_features = df.loc[:, features]
    rank_matrix = df_top_features.rank(axis=0)
    n = rank_matrix.shape[0]
    transformed_matrix = (n - rank_matrix + 1) / n
    edge_list = transformed_matrix.reset_index().melt(id_vars='sample.ID', var_name='Feature', value_name='Weight')
    edge_list.to_csv(f'../data/edg/edge_list_fold_{fold}.csv', index=False, header=False)
