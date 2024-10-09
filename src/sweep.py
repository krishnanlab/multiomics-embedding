'''
Author: Keenan Manpearl
Date: 2024-09-09

This script performs n2v+ embedding
and trains a diet and timepoint node classifier
for each of the 5 models.
as part of a wandb hyperparameter sweep

'''
import os 
import wandb
from argparse import ArgumentParser
from data import MultiomicsEmbedding
from model import cv, train, eval, log_model, log_final_metrics


def main(p, q, gamma, random_seed, n2v_mode, sweep_name):
    with wandb.init(project=sweep_name): 
        scores = {'time': [], 'diet': []}
        for model in range(1,6):
            data = MultiomicsEmbedding(model, p, q, gamma, random_seed, n2v_mode)
            for classifier_type in ['time', 'diet']:
                dataset = data.datasets[classifier_type]
                params, avg_score = cv(dataset, classifier_type, model, random_seed)
                log_model(classifier_type, model, params)
                scores[classifier_type].append(avg_score)
                classifier = train(dataset, params, random_seed)
                eval(classifier, dataset['train_data'], dataset['train_labels'], 'train', classifier_type, model)
                eval(classifier, dataset['test_data'], dataset['test_labels'], 'test', classifier_type, model)
        log_final_metrics(scores)
    wandb.finish()
    os.system("rm -f core.*")

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
    
    args = parser.parse_args()

    seed = args.seed
    n2v_mode = args.n2v
    p = args.p
    q = args.q
    g = args.g
    sweep_name = args.sweep
    main(p, q, g, seed, n2v_mode, sweep_name)