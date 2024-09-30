'''
Author: Keenan Manpearl
Date: 2024-08-29

This script performs feature selection and normalization.
Then creates an edge list. 
'''

import os
from typing import Union, Optional
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt


# threshold for feature selection
# higher tau score indicates a feature is specfific to a subset of samples
# a lower score indicates a more uniform distribution of feature across all samples
TAU_MIN = 0.8
# minimum and maximum missingness thresholds
# from adelle
MISSINGNESS_MIN = 0.2
MISSINGNESS_MAX = 0.75



def load_data(fp):
    '''
    load raw microbe/metabolite data
    '''
    df = pd.read_csv(fp)
    df.index = df['sample.ID']
    return df.drop(columns=['Unnamed: 0', 'sample.ID'])


def calculate_tau(feature: pd.Series):
    '''
    calculates tau score for a given feature
    '''
    x_i = feature / feature.max()
    n = len(feature)
    return sum(1 - x_i) / (n - 1)

def calculate_missingness(feature: pd.Series):
    '''
    calculates missingness score for a given feature
    '''
    return sum(feature == 0) / len(feature)


def calculate_scores(samples: list, 
                     data: pd.DataFrame, 
                     model: Union[str, int], 
                     filter: str):
    '''
    calculates tau or missingness scores for all features
    if model != full, only uses samples in the training set
    for the given model num
    '''
    if model != 'full':
        data = data.loc[samples]
    if filter == 'tau':
        scores = data.apply(calculate_tau)
    elif filter == 'missingness':
        scores = data.apply(calculate_missingness)
    scores.to_csv(f'../results/{filter}/scores_{model}.csv')
    plot_scores(scores, model, filter)
    return scores


def plot_scores(scores: pd.DataFrame, 
                label: Union[int, str], 
                filter: str):
    '''
    plots scores (tau or missingness) as a histogram
    label is either full or the cv fold used 
    '''
    scores.hist()
    plt.xlabel(f'{filter}')
    plt.ylabel('Frequency')
    plt.title(f'Dataset: {label}')
    plt.savefig(f'../results/{filter}/histogram_{label}.png')
    plt.close()

def select_top_features(scores: pd.Series, 
                        data: pd.DataFrame, 
                        min_threshold: Optional[int] = None, 
                        max_threshold: Optional[int] = None):
    '''
    select microbe/metabolite features where 
    score (tau or missingness) is above a minimum threshold
    and/or below a maximum threshold
    '''
    if min_threshold is not None:
        scores = scores[scores > min_threshold]
    if max_threshold is not None:
        scores = scores[scores < max_threshold]
    if min_threshold is None and max_threshold is None:
        raise ValueError("At least one of min_threshold or max_threshold must be specified.")
    
    return data.loc[:, scores.index.tolist()]

def get_feature_types(features: pd.DataFrame):
    '''
    get the type of each feature column (microbe or metabolite)
    '''
    metabolites = [col for col in features.columns if '_' in col]
    microbes = [col for col in features.columns if col not in metabolites]
    return microbes, metabolites

def feature_selection(data: pd.DataFrame,
                       filter: str, 
                       dataset: int, 
                       split: Optional[str] = None):
    '''
    perform feature selection on data
    using either tau or missingness (filter)
    dataset either specifies 'full' to use all samples
    or a specific model num to use only training samples.
    If using training samples splits is used to filter samples
    '''

    if filter == 'tau':
        top_features = select_top_features(scores=calculate_scores(split, data, dataset, filter),
                                            data=data, 
                                            min_threhsold=TAU_MIN, 
                                            max_threhsold=None)
    elif filter == 'missingness':
        scores = calculate_scores(split, data, dataset, filter)
        microbes, metabolites = get_feature_types(data)
        top_microbes = select_top_features(scores=scores[microbes],
                                            data=data.loc[:, microbes], 
                                            min_threshold=MISSINGNESS_MIN, 
                                            max_threshold=MISSINGNESS_MAX)
        top_metabolites = select_top_features(scores=scores[metabolites],
                                            data=data.loc[:, metabolites], 
                                            min_threshold=None, 
                                            max_threshold=MISSINGNESS_MAX)
        top_features = pd.concat([top_microbes, top_metabolites], axis=1)
        
    else:
        raise ValueError("Invalid filter. Must be 'tau' or 'missingness'")
    return top_features

def rank_normalization(data: pd.DataFrame):
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

def create_edge_list(matrix: pd.DataFrame, 
                     filter: Union[str, None], 
                     label: Union[str, int]):
    '''
    convert matrix to edge list and save as three column tsv
    filter is used as the directory name within edg
    and label is either full if the full dataset was used
    or the fold number if a specific CV fold was used
    '''
    if filter is None:
        filter = 'all_features'
    edge_list = matrix.reset_index().melt(id_vars='sample.ID', var_name='Feature', value_name='Weight')
    edge_list = edge_list[edge_list['Weight'] > 0]
    edge_list.to_csv(f'../data/edg/{filter}/edge_list_{label}.tsv', index=False, header=False, sep='\t')

def create_network(filter: Union[str, None], 
                    label: Union[str, int], 
                   splits: Union[pd.DataFrame, None]):
    '''
    loads raw data
    applies feature selection if specied in filter
    label specifies full to use all samples for feature selection
    or a cv fold number to use only training samples
    splits is the sample breakdown for each model
    it is only used 
    '''
    if filter == 'from_adelle':
        pass
    else: 
        # microbe/metabolite data
        data = load_data('../data/from_adelle/unfiltered_micro_metab.csv')
        if filter == 'tau' or filter == 'missingness':
            # apply filter
            features = feature_selection(data, filter, label, splits)
        else:
            features = data
        # get rank normalized features
        transformed_matrix = rank_normalization(features)
        # convert matrix to edge list and save
        create_edge_list(transformed_matrix, filter, label)


def get_training_samples(splits):
    for model in splits['run'].unique():
        samples = splits[(splits['run'] == model) & (splits['partition'] == 'train')]['nodes']
        yield (model, samples)

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("--filter",
                        help="filtering method if feature selection is desired (tau or missingness)",
                        required=False,
                        type=str,
                        choices = ['tau', 'missingness', 'from_adelle', 'all_features'],
                        default=None)
    parser.add_argument("--set",
                        help="whether to use training or full dataset for feature selection",
                        required=False,
                        type=str,
                        choices = ['train', 'full'],
                        default=None)
    
    args = parser.parse_args()
    filter = args.filter
    set = args.set

    if filter == 'tau' or filter == 'missingness':
        out_results = f'../results/{filter}'
        if not os.path.exists(out_results):
            os.makedirs(out_results)
    out_edge = f'../data/edg/{filter}'
    if not os.path.exists(out_edge):
        os.makedirs(out_edge)

    # train/test splits for each model
    splits = pd.read_csv('../data/from_adelle/sample_breakdown.csv')
    
    if set == 'train':
        [create_network(filter, model, split) for model, split in get_training_samples(splits)]
    else:
        create_network(filter, set, splits)
            
