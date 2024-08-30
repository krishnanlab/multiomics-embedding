'''
Author: Keenan Manpearl
Date: 2024-08-29

This script performs n2v+ embeddig with pecanpy 
and trains a node classifier to predict baseline vs endpoint samples.

'''

import pandas as pd

from argparse import ArgumentParser 
from pecanpy import pecanpy as node2vec


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("-fold",
                        help="required to load proper edge list and labels",
                        required=True,
                        type=int)
    parser.add_argument("-p",
                        help="pecanpy p parameter: 0 <= p <= 1",
                        required=True,
                        type=float)
    parser.add_argument("-q",
                        help="pecanpy q parameter: 0 <= q <= 1",
                        required=True,
                        type=float)
    parser.add_argument("-g",
                        help="pecanpy gamma parameter: 0 <= g <= 1",
                        required=True,
                        type=float)
    parser.add_argument("-reg",
                        help="regularization parameter for logistic regression",
                        required=True,
                        type=float)
    parser.add_argument("-seed",
                        help="seed for reproducibility",
                        required=False,
                        type=int,
                        default=42)
    
    
    args = parser.parse_args()

    fold = str(args.fold)
    p = args.p
    q = args.q
    gamma = args.g
    reg = args.reg
    seed = args.seed

    splits = pd.read_csv('../data/from_adelle/sample_breakdown.csv')
    splits = splits[splits['run'] == fold]
    splits['Time'] = splits['Time'].map({'Baseline': 0, 'Endpoint': 1})

    train_ids = splits[splits['partition'] == 'train']['nodes']
    test_ids = splits[splits['partition'] == 'test']['nodes']
    train_labels = splits[splits['partition'] == 'train']['Time']
    test_labels = splits[splits['partition'] == 'test']['Time']

    edg_file = f'../data/edg/edge_list_fold_{fold}.csv'

    g = node2vec.PreComp(p=p, q=q, workers=1, verbose=True, extend=True, gamma=gamma,random_state=seed)
    g.read_edg(edg_file, weighted=True, directed=False,delimiter=',')
    emb = g.embed()
    print(emb)


