'''
Author: Keenan Manpearl
Date: 2024-08-29

This script performs n2v+ embeddig with pecanpy 
and trains a node classifier to predict baseline vs endpoint samples.

'''
import os
import wandb
import pandas as pd

from argparse import ArgumentParser 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from pecanpy import pecanpy as node2vec


def get_idx(file):
    with open(file, 'r') as f:
        idx = f.readlines()
    return [i.strip() for i in idx]

def load_data(df, splits, idx):
    labels = splits.loc[idx]['Time']
    data = df.loc[idx]
    return labels, data


def embed_network(fold, p, q, gamma, seed):
    edg_file = f'../data/edg/edge_list_fold_{fold}.csv'
    g = node2vec.SparseOTF(p=p, q=q, workers=4, verbose=True, extend=True, gamma=gamma, random_state=seed)
    g.read_edg(edg_file, weighted=True, directed=False,delimiter=',')
    nodes = g.nodes
    emb = g.embed()
    df = pd.DataFrame(emb,index=nodes)
    df.to_csv(f'../data/emb/{fold}/emb_p_{p}_q_{q}_g_{gamma}.tsv',sep='\t')
    return df

def score_model(scores, split, cv_fold, labels, predictions):
    scores[f'{split}_accuracy'][cv_fold] = accuracy = accuracy_score(labels, predictions)
    scores[f'{split}_balanced_accuracy'][cv_fold] = balanced_accuracy = balanced_accuracy_score(labels, predictions)
    scores[f'{split}_precision'][cv_fold] = precision = precision_score(labels, predictions)
    scores[f'{split}_recall'][cv_fold] = recall = recall_score(labels, predictions)
    scores[f'{split}_f1'][cv_fold] = f1 =f1_score(labels, predictions)

    wandb.summary[f'{split}_accuracy_{cv_fold}'] = accuracy
    wandb.summary[f'{split}_balanced_accuracy_{cv_fold}'] = balanced_accuracy
    wandb.summary[f'{split}_precision_{cv_fold}'] = precision
    wandb.summary[f'{split}_recall_{cv_fold}'] = recall
    wandb.summary[f'{split}_f1_{cv_fold}'] = f1

    return scores

def aggregate_scores(scores):
    for metric in scores.keys():
        avg = 0
        for score in scores[metric].values():
            avg += score
        scores[metric]['avg'] = avg_score = avg / 5
        wandb.summary[f'{metric}_avg'] = avg_score

    return scores



if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("-fold",
                        help="required to load proper edge list and labels",
                        required=True,
                        type=int)
    parser.add_argument("-p",
                        help="node2vec p parameter: 0 <= p <= 1",
                        required=True,
                        type=float)
    parser.add_argument("-q",
                        help="node2vec q parameter: 0 <= q <= 1",
                        required=True,
                        type=float)
    parser.add_argument("-g",
                        help="node2vec gamma parameter: 0 <= g <= 1",
                        required=True,
                        type=float)
    parser.add_argument("-reg",
                        help="inverse of regularization strength for logistic regression",
                        required=True,
                        type=int)
    parser.add_argument("-penalty",
                        help="regularization type for logistic regression",
                        required=False,
                        type=str,
                        default='l2')
    parser.add_argument("-seed",
                        help="seed for reproducibility",
                        required=False,
                        type=int,
                        default=42)
    
    
    args = parser.parse_args()

    fold = args.fold
    seed = args.seed
    params = {'p' : args.p,
              'q' : args.q,
              'gamma' : args.g,
              'penalty' : args.penalty,
              # inverse of regularization strength
              # smaller number == stronger regularization
              'c' : args.reg}

    os.makedirs(f'../data/emb/{fold}',exist_ok=True)
    os.makedirs(f'../results/model_tuning/{fold}', exist_ok=True)

    splits = pd.read_csv('../data/from_adelle/sample_breakdown.csv')
    splits = splits[splits['run'] == fold]
    splits['Time'] = splits['Time'].map({'Baseline': 0, 'Endpoint': 1})
    splits.index = splits['nodes']

    with wandb.init(project="multiomics", config=params):
        config = wandb.config
        p = config.p
        q = config.q
        gamma = config.gamma
        c = config.c
        penalty = config.penalty

        df = embed_network(fold, p, q, gamma, seed)

        scores = {'train_accuracy': {}, 
                'test_accuracy': {}, 
                'train_balanced_accuracy': {}, 
                'test_balanced_accuracy': {}, 
                'train_precision': {}, 
                'test_precision': {}, 
                'train_recall': {}, 
                'test_recall': {}, 
                'train_f1': {}, 
                'test_f1': {}
                }
        
        for cv_fold in [0,1,2,3,4]:
            train_idx = get_idx(f'../data/cv_folds/{fold}/train_{cv_fold}.txt')
            test_idx = get_idx(f'../data/cv_folds/{fold}/test_{cv_fold}.txt')
            train_labels, train_data = load_data(df, splits, train_idx)
            test_labels, test_data = load_data(df, splits, test_idx)
            
            logreg = LogisticRegression(C=c, penalty=penalty)
            logreg.fit(train_data, train_labels)
            train_predictions = logreg.predict(train_data)
            test_predictions = logreg.predict(test_data)
            test_probabilities = logreg.predict_proba(test_data)
            wandb.sklearn.plot_roc(test_labels, test_probabilities, ['Baseline', 'Endpoint'])
            wandb.sklearn.plot_precision_recall(test_labels, test_probabilities, ['Baseline', 'Endpoint'])
            wandb.sklearn.plot_confusion_matrix(test_labels, test_predictions, ['Baseline', 'Endpoint'])
            wandb.sklearn.plot_summary_metrics(logreg, train_data, train_labels, test_data, test_labels)

            scores = score_model(scores, 'train', cv_fold, train_labels, train_predictions)
            scores = score_model(scores, 'test', cv_fold, test_labels, test_predictions)

        scores = aggregate_scores(scores)
        pd.DataFrame(scores).to_csv(f'../results/model_tuning/{fold}/p_{p}_q_{q}_g_{gamma}_c_{c}.tsv',sep='\t')


