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
    '''
    calculates tau score for a given feature
    '''
    x_i = feature / feature.max()
    n = len(feature)
    return sum(1 - x_i) / (n - 1)


def calculate_tau_scores(splits, data):
    '''
    calculates tau scores for all features
    only uses samples in the training set
    higher score = more variability across samples
    '''
    samples = splits[(splits['run'] == fold) & (splits['partition'] == 'train')]['nodes']
    subset = data.loc[samples]
    tau_scores = subset.apply(calculate_tau)
    tau_scores.to_csv(f'../results/tau_scores/tau_scores_fold_{fold}.csv')
    return(tau_scores)

def plot_tau_scores(tau_scores, fold):
    '''
    plots tau scores
    '''
    tau_scores.hist()
    plt.xlabel('Tau Scores')
    plt.ylabel('Frequency')
    plt.title(f'Fold {fold}')
    plt.savefig(f'../results/tau_scores/tau_scores_histogram_fold_{fold}.png')
    plt.close()

def select_top_features(tau_scores, threshold, data):
    '''
    select microbe/metabolite features where tau > threshold
    '''
    features = tau_scores[tau_scores > threshold].index.tolist()
    return data.loc[:, features]

def rank_normalization(data):
    '''
    convert raw abundance data to rank normalized
    zero values remain as zero
    
    dat = pd.DataFrame({'feature 1': [0,1,2,3],
                   'feature2': [0,0,1,0],
                   'feature 3': [10,1,3,3]})
    normalized = rank_normalization(dat)

        feature 1  feature2  feature 3
    0       0.00       0.0      1.000
    1       0.50       0.0      0.250
    2       0.75       1.0      0.625
    3       1.00       0.0      0.625
    '''
    rank_matrix = data.rank(axis=0, ascending=False)
    n = rank_matrix.shape[0]
    rank_matrix = (n - rank_matrix + 1) / n
    rank_matrix[data == 0] = 0
    return rank_matrix

def create_edge_list(matrix, fold):
    '''
    convert matrix to edge list and save as three column tsv
    '''
    edge_list = matrix.reset_index().melt(id_vars='sample.ID', var_name='Feature', value_name='Weight')
    edge_list.to_csv(f'../data/edg/edge_list_fold_{fold}.tsv', index=False, header=False, sep='\t')

def load_data():
    '''
    load raw microbe/metabolite data
    '''
    df = pd.read_csv('../data/from_adelle/unfiltered_micro_metab.csv')
    df.index = df['sample.ID']
    return df.drop(columns=['Unnamed: 0', 'sample.ID'])



if __name__ == '__main__':
    # train/test splits for each fold 
    splits = pd.read_csv('../data/from_adelle/sample_breakdown.csv')
    # microbe/metabolite data
    data = load_data()

    for fold in splits['run'].unique():
        # calculae tau scores considering only samples in training set for each fold
        tau_scores = calculate_tau_scores(splits, data)
        plot_tau_scores(tau_scores, fold)
        # select features where tau score > threshold
        top_features = select_top_features(tau_scores, TAU_THRESHOLD, data)
        # rank normalize features
        transformed_matrix = rank_normalization(top_features)
        # convert matrix to edge list and save
        create_edge_list(transformed_matrix, fold)
