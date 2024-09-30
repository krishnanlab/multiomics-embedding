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
    Fold determins which original data fold to use.
    This impacts the feature selection.
    p, q, and gamma are node2vec parameters.
    n2v mode determins whether the embedding is done using 
    precomputed transtion probabilities, or on the fly
    this only effects performance. 
    Pre-comp is much faster but more memory intensive 
    (Got OOM with 256GB allocated).
    SparseOTF is slower but uses less memory. 

    '''
    def __init__(self, 
                 fold: int, 
                 p: float, 
                 q: float, 
                 gamma: float, 
                 seed: int, 
                 n2v_mode: str, 
                 filter: str):
        self.fold = fold
        self.p = p
        self.q = q
        self.gamma = gamma
        self.seed = seed
        self.nv2_mode = n2v_mode
        self.filter = filter
        self.load_labels()
        self.load_embedding()
        self.setup_cv_datasets()
        self.setup_test_data()

    def load_labels(self):
        '''
        Convert time points into binary labels for ML
        '''
        labels = pd.read_csv('data/raw/sample_breakdown.csv')
        labels = labels[labels['run'] == self.fold]
        labels['Time'] = labels['Time'].map({'Baseline': 0, 'Endpoint': 1})
        labels.index = labels['nodes']
        self.labels = labels
    
    def embed_network(self, emb_file: str):
        '''
        load the edge list and create a node2vec+ embedding
        '''
        if self.filter != 'from_adelle':
            edg_file = f'data/edg/{self.filter}/edge_list_full.tsv'
        else:
            edg_file = 'data/edg/from_adelle/full_data_pecan_3.tsv'
        if self.nv2_mode == 'OTF':
            print('Embedding network using SparseOTF')
            g = node2vec.SparseOTF(p=self.p, q=self.q, workers=4, verbose=True, extend=True, gamma=self.gamma, random_state=self.seed)
            g.read_edg(edg_file, weighted=True, directed=False)
        elif self.nv2_mode == 'Pre':
            print('Embedding network using PreComp')
            g = node2vec.PreComp(p=self.p, q=self.q, workers=4, verbose=True, extend=True, gamma=self.gamma, random_state=self.seed)
            g.read_edg(edg_file, weighted=True, directed=False)
            g.preprocess_transition_probs()
        nodes = g.nodes
        emb = g.embed()
        df = pd.DataFrame(emb,index=nodes)
        df.to_csv(emb_file,sep='\t')
        self.emb = df

    def load_embedding(self):
        '''
        Check if an embedding file with specified parameters
        exits for the fold.
        If not, create one. 
        '''
        p = self.p
        q = self.q
        gamma = self.gamma
        filter = self.filter
        root = f'data/emb/{filter}'
        emb_file = f'{root}/emb_p_{p}_q_{q}_g_{gamma}.tsv'
        if os.path.exists(emb_file):
            print(f'Loading embedding from file')
            self.emb = pd.read_csv(emb_file, sep='\t', index_col=0)
        else:
            os.makedirs(root,exist_ok=True)  
            self.embed_network(emb_file)
            
    
    def get_cv_idx(self, split: str, cv_fold: int):
        '''
        Load the proper CV split indices.
        '''
        if split == 'val':
            split = 'test'
        with open(f'data/cv_folds/{self.fold}/{split}_{cv_fold}.txt', 'r') as f:
            idx = f.readlines()
        return [i.strip() for i in idx]

    def setup_cv_datasets(self):
        '''
        For a specified fold, 
        create 5-fold CV datasets for hyperparameter tuning.
        The folds are held constant across all models
        get_idx retrieves the proper train/val  indices for each CV.
        '''
        self.datasets = {}
        for cv_fold in range(NUM_CV_FOLDS):
            self.datasets[cv_fold] = {}
            for split in ['train', 'val']:
                idx = self.get_cv_idx(split, cv_fold)
                self.datasets[cv_fold][f'{split}_labels'] = self.labels.loc[idx]['Time'].values
                self.datasets[cv_fold][f'{split}_data'] =  self.emb.loc[idx].values

    def setup_test_data(self):
        '''
        Set up the test data for the given fold.
        This is the final test set and should NOT be used for hyperparameter tuning.

        '''
        test_idx = self.labels[self.labels['partition'] == 'test'].index
        train_idx = self.labels[self.labels['partition'] == 'train'].index
        self.test_dataset = {'test_labels': self.labels.loc[test_idx]['Time'].values,
                             'test_data': self.emb.loc[test_idx].values,
                             'train_labels': self.labels.loc[train_idx]['Time'].values,
                             'train_data': self.emb.loc[train_idx].values}
    



    