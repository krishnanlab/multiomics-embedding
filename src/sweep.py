'''
Author: Keenan Manpearl
Date: 2024-09-09

This script performs n2v+ embedding
and trains a node classifier
as part of a wandb hyperparameter sweep

'''
import wandb
from argparse import ArgumentParser
from data import MultiomicsEmbedding
from train import Trainer



def main(fold, p, q, g, penalty, c, seed, n2v_mode, filter):
    with wandb.init(): 
        data = MultiomicsEmbedding(fold, p, q, g, seed, n2v_mode, filter)
        trainer = Trainer(penalty, c)
        trainer.train(data)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--fold",
                        help="required to load proper edge list and labels",
                        required=True,
                        type=int)
    parser.add_argument("--p",
                        help="node2vec p parameter",
                        required=True,
                        type=float)
    parser.add_argument("--q",
                        help="node2vec q parameter",
                        required=True,
                        type=float)
    parser.add_argument("--g",
                        help="node2vec gamma parameter",
                        required=True,
                        type=int)
    parser.add_argument("--c",
                        help="inverse of regularization strength for logistic regression",
                        required=True,
                        type=float)
    parser.add_argument("--penalty",
                        help="regularization type for logistic regression",
                        required=False,
                        type=str,
                        default='l2')
    parser.add_argument("--seed",
                        help="seed for reproducibility",
                        required=False,
                        type=int,
                        default=42)
    parser.add_argument("--n2v",
                        help="node2vec graph type: effects time and memory usage",
                        required=False,
                        type=str,
                        choices = ['OTF', 'Pre'],
                        default='Pre')
    parser.add_argument("--filter",
                        help="filtering method for feature selection",
                        required=True,
                        type=str,
                        choices = ['tau', 'missingness'])
    
    args = parser.parse_args()

    fold = args.fold
    seed = args.seed
    n2v_mode = args.n2v
    p = args.p
    q = args.q
    g = args.g
    penalty = args.penalty
    filter = args.filter
    # inverse of regularization strength
    # smaller number == stronger regularization
    c = args.c
    main(fold, p, q, g, penalty, c, seed, n2v_mode, filter)