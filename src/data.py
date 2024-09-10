'''
Author: Keenan Manpearl
Date: 2024-09-09

This script performs n2v+ embeddig with pecanpy 
and creates cross-validation datasets

'''

import pandas as pd
from pecanpy import pecanpy as node2vec
import os

NUM_CV_FOLDS = 5

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
    def __init__(self, fold, p, q, gamma, seed, n2v_mode):
        self.fold = fold
        self.p = p
        self.q = q
        self.gamma = gamma
        self.seed = seed
        self.nv2_mode = n2v_mode
        self.load_labels()
        self.load_embedding()
        self.setup_cv_datasets()

    def load_labels(self):
        '''
        Convert timepoints into binary labels for ML
        '''
        labels = pd.read_csv('data/from_adelle/sample_breakdown.csv')
        labels= labels[labels['run'] == self.fold]
        labels['Time'] = labels['Time'].map({'Baseline': 0, 'Endpoint': 1})
        labels.index = labels['nodes']
        self.labels = labels
    
    def embed_network(self, emb_file):
        '''
        load the edge list and create a node2vec+ embedding
        '''
        p = self.p
        q = self.q
        gamma = self.gamma
        seed = self.seed
        fold = self.fold 
        edg_file = f'data/edg/edge_list_fold_{fold}.csv'
        if self.nv2_mode == 'OTF':
            print('Embedding network using SparseOTF')
            g = node2vec.SparseOTF(p=p, q=q, workers=4, verbose=True, extend=True, gamma=gamma, random_state=seed)
            g.read_edg(edg_file, weighted=True, directed=False, delimiter=',')
        elif self.nv2_mode == 'Pre':
            print('Embedding network using PreComp')
            g = node2vec.PreComp(p=p, q=q, workers=4, verbose=True, extend=True, gamma=gamma, random_state=seed)
            g.read_edg(edg_file, weighted=True, directed=False, delimiter=',')
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
        fold = self.fold    
        emb_file = f'data/emb/{fold}/emb_p_{p}_q_{q}_g_{gamma}.tsv'
        if os.path.exists(emb_file):
            print(f'Loading embedding from file')
            self.emb = pd.read_csv(emb_file, sep='\t', index_col=0)
        else:
            self.embed_network(emb_file)
            
    
    def get_idx(self, split, cv_fold):
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
        get_idx retrieves the proper indices for each CV.
        '''
        self.datasets = {}
        for cv_fold in range(NUM_CV_FOLDS):
            self.datasets[cv_fold] = {}
            for split in ['train', 'val']:
                idx = self.get_idx(split, cv_fold)
                self.datasets[cv_fold][f'{split}_labels'] = self.labels.loc[idx]['Time'].values
                self.datasets[cv_fold][f'{split}_data'] =  self.emb.loc[idx]


        
    



    