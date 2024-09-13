'''
Author: Keenan Manpearl
Date: 2024-08-29

This script performs feature selection and normalization.
Then creates an edge list. 
'''

import os
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt


# threshold for feature selection
# higher tau score indicates a feature is specfific to a subset of samples
# a lower score indicates a more uniform distribution of feature across all samples
TAU_THRESHOLD = 0.8
# minimum and maximum missingness thresholds
# from adelle
MISSINGNESS_MIN = 0.1
MISSINGNESS_MAX = 0.75



def load_data():
    '''
    load raw microbe/metabolite data
    '''
    df = pd.read_csv('../data/from_adelle/unfiltered_micro_metab.csv')
    df.index = df['sample.ID']
    return df.drop(columns=['Unnamed: 0', 'sample.ID'])

def calculate_tau(feature):
    '''
    calculates tau score for a given feature
    '''
    x_i = feature / feature.max()
    n = len(feature)
    return sum(1 - x_i) / (n - 1)

def calculate_missingness(feature):
    '''
    calculates missingness score for a given feature
    '''
    return sum(feature == 0) / len(feature)


def calculate_scores(splits, data, fold, filter):
    '''
    calculates tau or missingness scores for all features
    only uses samples in the training set
    higher score = more variability across samples
    '''
    if fold != 'full':
        samples = splits[(splits['run'] == fold) & (splits['partition'] == 'train')]['nodes']
        data = data.loc[samples]
    if filter == 'tau':
        scores = data.apply(calculate_tau)
    elif filter == 'missingness':
        scores = data.apply(calculate_missingness)
    scores.to_csv(f'../results/{filter}/scores_{fold}.csv')
    return scores


def plot_scores(scores, fold, filter):
    '''
    plots scores (tau or missingness) as a histogram
    '''
    scores.hist()
    plt.xlabel(f'{filter}')
    plt.ylabel('Frequency')
    plt.title(f'Dataset: {fold}')
    plt.savefig(f'../results/{filter}/histogram_{fold}.png')
    plt.close()

def select_top_features(scores, data, min_threshold, max_threshold=None):
    '''
    select microbe/metabolite features where 
    score (tau or missingness) is above a minimum threshold
    and optionally below a maximum threshold
    '''
    features = scores[scores > min_threshold]
    if max_threshold is not None:
        features = features[features < max_threshold]
    return data.loc[:, features.index.tolist()]

def feature_selection(data, filter, fold, splits=None):
    '''
    perform feature selection on data
    using either tau or missingness (filter)
    fold either specifies 'full' to use all samples
    or a specific fold to use only training samples.
    If using training samples splits is used to filter samples
    '''
    # Define the computation and filtering logic based on the filter_type
    if filter == 'tau':
        threshold_min, threshold_max = TAU_THRESHOLD, None
    elif filter == 'missingness':
        threshold_min, threshold_max = MISSINGNESS_MIN, MISSINGNESS_MAX
    else:
        raise ValueError("Invalid filter_type. Must be 'tau' or 'missingness'.")
    # Calculate tau or missingness scores
    scores = calculate_scores(splits, data, fold, filter)
    # Plot the scores (either tau or missingness)
    plot_scores(scores, fold, filter)
    # Select features based on thresholds
    top_features = select_top_features(scores, data, threshold_min, threshold_max)
    return top_features

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

def create_edge_list(matrix, filter, fold):
    '''
    convert matrix to edge list and save as three column tsv
    '''
    if filter is None:
        filter = 'all_features'
    edge_list = matrix.reset_index().melt(id_vars='sample.ID', var_name='Feature', value_name='Weight')
    edge_list = edge_list[edge_list['Weight'] > 0]
    edge_list.to_csv(f'../data/edg/{filter}/edge_list_{fold}.tsv', index=False, header=False, sep='\t')

def create_network(data, filter, fold, splits):
    if filter is not None:
        # apply filter
        features = feature_selection(data, filter, fold, splits)
    else:
        features = data
    # get rank normalized features
    transformed_matrix = rank_normalization(features)
    # convert matrix to edge list and save
    create_edge_list(transformed_matrix, filter, fold)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--filter",
                        help="filtering method if feature selection is desired (tau or missingness)",
                        required=False,
                        type=str,
                        choices = ['tau', 'missingness'],
                        default=None)
    parser.add_argument("--set",
                        help="whether to use training or full dataset for feature selection",
                        required=False,
                        type=str,
                        choices = ['train', 'full'],
                        default='train')
    
    args = parser.parse_args()
    filter = args.filter
    set = args.set

    if filter is not None:
        out_results = f'../results/{filter}'
        if not os.path.exists(out_results):
            os.makedirs(out_results)
        out_edge = f'../data/edg/{filter}'
        if not os.path.exists(out_edge):
            os.makedirs(out_edge)
    else: 
        out_edge = f'../data/edg/all_features'
        if not os.path.exists(out_edge):
            os.makedirs(out_edge)

    # train/test splits for each fold 
    splits = pd.read_csv('../data/from_adelle/sample_breakdown.csv')
    # microbe/metabolite data
    data = load_data()
    
    if set == 'train':
        [create_network(data, filter, fold, splits) for fold in splits['run'].unique()]
    else:
        create_network(data, filter, set, splits)
            
