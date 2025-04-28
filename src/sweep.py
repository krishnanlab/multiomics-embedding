'''
Author: Keenan Manpearl
Date: 2024-09-09

This script performs n2v+ embedding
and trains a diet and timepoint node classifier
for each of the 5 models.
as part of a wandb hyperparameter sweep

'''

import wandb
from argparse import ArgumentParser
from model import train_loop


def main(p, q, gamma, random_seed, n2v_mode, sweep_name, save):
    with wandb.init(project=sweep_name): 
        train_loop(p, q, gamma, random_seed, n2v_mode, save)
    wandb.finish()

if __name__ == '__main__':
    parser = ArgumentParser()
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
                        default='OTF')
    parser.add_argument("--sweep",
                        help="sweep name",
                        required=True,
                        type=str)
    parser.add_argument("--save",
                        help="whether to save the model",
                        required=False,
                        type=bool,
                        default=False)
    
    args = parser.parse_args()

    seed = args.seed
    n2v_mode = args.n2v
    p = args.p
    q = args.q
    g = args.g
    sweep_name = args.sweep
    save = args.save
    main(p, q, g, seed, n2v_mode, sweep_name, save)