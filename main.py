'''
Author: Keenan Manpearl
Date: 2024-09-09

This script performs n2v+ embedding
and trains a node classifier.
Logging is done with wandb.

'''

import wandb
import os
from argparse import ArgumentParser
from src.data import MultiomicsEmbedding
from src.model import train_loop


def main(random_seed, param_dict, project_name, n2v_mode, save):
    with wandb.init(project=project_name, config=param_dict):
        config = wandb.config
        p = config.p
        q = config.q
        gamma = config.gamma
        train_loop(p, q, gamma, random_seed, n2v_mode, save)
    wandb.finish()
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--name",
                        help="wandb project name",
                        required=False,
                        type=str,
                        default='multiomics_joint_classifiers')
    parser.add_argument("--p",
                        help="node2vec+ p (in out) parameter",
                        required=True,
                        type=float)
    parser.add_argument("--q",
                        help="node2vec+ q (return) parameter",
                        required=True,
                        type=float)
    parser.add_argument("--g",
                        help="node2vec+ gamma parameter",
                        required=True,
                        type=int)
    parser.add_argument("--seed",
                        help="random seed for reproducibility",
                        required=False,
                        type=int,
                        default=42)
    parser.add_argument("--n2v",
                        help="node2vec graph type: effects time and memory usage",
                        required=False,
                        type=str,
                        choices = ['OTF', 'Pre'],
                        default='OTF')
    parser.add_argument("--save",
                        help="whether to save the model",
                        required=False,
                        type=bool,
                        default=False)
    
    args = parser.parse_args()

    name = args.name
    seed = args.seed
    params = {'p' : args.p,
              'q' : args.q,
              'gamma' : args.g}
    n2v = args.n2v
    save = args.save
    main(seed, params, name, n2v, save)


