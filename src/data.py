'''
Author: Keenan Manpearl
Date: 2024-09-09

This script performs n2v+ embeddig with pecanpy 
and creates cross-validation datasets

'''
import os
import pandas as pd
from pecanpy import pecanpy as node2vec


NUM_CV_FOLDS = 10

class MultiomicsEmbedding():
    '''
    This class holds the labels and creates a network embedding. 
    Model number is used to load the proper labels.
    p, q, and gamma are node2vec parameters.
    n2v mode determins whether the embedding is done using 
    precomputed transtion probabilities, or on the fly
    this only effects performance. 
    Pre-comp is much faster but more memory intensive 
    (Got OOM with 256GB allocated).
    SparseOTF is slower but uses less memory. 

    '''
    def __init__(self, 
                 model: int, 
                 p: float, 
                 q: float, 
                 gamma: float, 
                 seed: int, 
                 n2v_mode: str):
        self.model = model
        self.p = p
        self.q = q
        self.gamma = gamma
        self.seed = seed
        self.nv2_mode = n2v_mode
        self.datasets = self.setup_data()

    def setup_embedding(self):
        '''
        Check if an embedding file with specified parameters exits.
        If not, create one. 
        '''
        p = self.p
        q = self.q
        gamma = self.gamma
        root = f'data/emb/'
        emb_file = f'{root}/emb_p_{p}_q_{q}_g_{gamma}.tsv'
        if os.path.exists(emb_file):
            print(f'Loading embedding from file')
            return pd.read_csv(emb_file, sep='\t', index_col=0)
        else:
            os.makedirs(root,exist_ok=True) 
            return embed_network(emb_file, self.nv2_mode, p, q, gamma, self.seed)  
    
    def setup_data(self):
        '''
        Set up the datasets for the given model
        '''
        emb = self.setup_embedding()
        labels = load_timepoint_labels(self.model)
        # get timepoint indexes
        test_time_idx = labels[labels['partition'] == 'test'].index
        train_time_idx = labels[labels['partition'] == 'train'].index
        # get diet indexes 
        test_diet_idx = [idx for idx in test_time_idx if 'End' in idx]
        train_diet_idx = [idx for idx in train_time_idx if 'End' in idx]
        diet_labels = load_diet_labels()
        return {'time' : {
            'train_data': emb.loc[train_time_idx].values,
            'test_data': emb.loc[test_time_idx].values,
            'train_labels': labels.loc[train_time_idx]['Label'].values,
            'test_labels': labels.loc[test_time_idx]['Label'].values 
            },
            'diet' : {
            'train_data': emb.loc[train_diet_idx].values,
            'test_data': emb.loc[test_diet_idx].values,
            'train_labels': diet_labels.loc[train_diet_idx]['Label'].values,
            'test_labels': diet_labels.loc[test_diet_idx]['Label'].values
            }}


def load_timepoint_labels(model: int):
    '''
    Convert time points into binary labels for ML
    '''
    labels = pd.read_csv('data/raw/sample_breakdown.csv')
    labels = labels[labels['run'] == model]
    labels['Label'] = labels['Time'].map({'Baseline': 0, 'Endpoint': 1})
    labels.index = labels['nodes']
    return labels

def load_diet_labels():
    '''
    Convert diet into binary labels for ML
    '''
    labels = pd.read_csv('data/raw/microbiome_info_data.csv', usecols=['sample.name', 'Group', 'Time'])
    labels.index = labels['sample.name']
    labels['Label'] = labels['Group'].map({'Dairy': 0, 'Meat': 1})
    labels = labels[~labels.index.duplicated(keep='first')]
    return labels


def embed_network(emb_file: str, nv2_mode: str, p: float, q: float, gamma: int, seed: int):
    '''
    load the edge list and create a node2vec+ embedding
    '''
    edg_file = 'data/edg/full_data_pecan_3.tsv'
    if nv2_mode == 'OTF':
        print('Embedding network using SparseOTF')
        g = node2vec.SparseOTF(p=p, q=q, workers=4, verbose=True, extend=True, gamma=gamma, random_state=seed)
        g.read_edg(edg_file, weighted=True, directed=False)
    elif nv2_mode == 'Pre':
        print('Embedding network using PreComp')
        g = node2vec.PreComp(p=p, q=q, workers=4, verbose=True, extend=True, gamma=gamma, random_state=seed)
        g.read_edg(edg_file, weighted=True, directed=False)
        g.preprocess_transition_probs()
    nodes = g.nodes
    emb = g.embed()
    df = pd.DataFrame(emb,index=nodes)
    df.to_csv(emb_file,sep='\t')
    return df
