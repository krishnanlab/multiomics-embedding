'''
Author: Keenan Manpearl
Date: 2024-09-09

This script trains a node classifier to predict 
baseline vs endpoint samples.

'''

import wandb
import numpy as np 
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


NUM_CV_FOLDS = 5

class LogReg:
    def __init__(self, penalty: str, c: float):
        solver = self.get_solver(penalty)
        self.model = LogisticRegression(penalty=penalty, C=c, solver=solver)
        self.initialize_metric_dict()

    def get_solver(self, penalty: str):
        '''
        determine the apropriate solver for the given penalty
        '''
        if penalty == 'l1':
            return 'liblinear'
        else:
            return 'lbfgs'
    
    def initialize_metric_dict(self):
        '''
        set up dictionary to log metrics
        train and val will hold one entry for each CV fold and one avg entry 
        test will hold one entry for the final test set
        '''
        self.metrics = {'accuracy': {'train': {}, 'val': {}, 'test': None},
                        'balanced_accuracy': {'train': {}, 'val': {}, 'test': None}, 
                        'precision': {'train': {}, 'val': {}, 'test': None}, 
                        'recall': {'train': {}, 'val': {}, 'test': None}, 
                        'f1': {'train': {}, 'val': {}, 'test': None} }
    
    def log_metrics(self, split: str, metrics: dict, cv_fold: Optional[int] = None):
        '''
        split specifies train, val, or test
        metrics is a dictionary of metric names and scores
        cv_fold is only provided if logging during cv training

        logs the metrics to self.metrics and wandb
        if cv_fold is provided (for train or val)
        then logs metrics to self.metrics[split][cv_fold]
        if not then logs to self.metrics[split]
        '''
        if cv_fold is not None:
            for metric, score in metrics.items():
                self.metrics[metric][split][cv_fold] = score
                wandb.summary[f'{metric}_{split}_{cv_fold}'] = score
        else:
            for metric, score in metrics.items():
                self.metrics[metric][split] = score
                wandb.summary[f'{metric}_{split}'] = score
        

    
    def eval_cv(self, dataset: dict, split: str, cv_fold: int):
        '''
        for a given cv fold, generate predictions 
        and log model metrics
        dataset is the dictionary of train/val data and labels
        for a given fold
        split specifices train or val
        '''
        data = dataset[f'{split}_data']
        labels = dataset[f'{split}_labels']
        preds = self.model.predict(data)
        scores = score_model(labels, preds)
        self.log_metrics(split, scores, cv_fold)

    
    def log_aggregate_scores(self):
        '''
        average all metrics across cv folds and log to wandb
        '''
        for metric in self.metrics.keys():
            for split in ['train', 'val']:
                avg = 0
                for score in self.metrics[metric][split].values():
                    avg += score
                self.metrics[metric][split]['avg'] = avg_score = avg / len(self.metrics[metric][split])
                wandb.summary[f'{metric}_{split}_avg'] = avg_score
                wandb.log({f'{metric}_{split}_avg': avg_score})

    def train(self, data):
        '''
        train the model on each training fold of the CV
        and evaluate on the validation fold 
        data should of class MultiomicsEmbedding
        '''
        for cv_fold in range(NUM_CV_FOLDS):
            # get train/test data and labels for each fold
            dataset = data.datasets[cv_fold]
            # train the model
            self.model.fit(dataset['train_data'], dataset['train_labels'])
            # generate and log metrics
            self.eval_cv(dataset, 'train', cv_fold)
            self.eval_cv(dataset, 'val', cv_fold)
        # average metrics across folds
        self.log_aggregate_scores()

    def eval(self, test_set: dict):
        '''
        retrain a final model on all training data
        and generate metrics for the held out test set  
        '''
        print(test_set)
        test_data = test_set['test_data']
        test_labels = test_set['test_labels']
        train_data = test_set['train_data']
        train_labels = test_set['train_labels']
        self.model.fit(train_data, train_labels)
        preds = self.model.predict(test_data)
        probs = self.model.predict_proba(test_data)
        scores = score_model(test_labels, preds)
        self.log_metrics('test', scores)

        wandb.sklearn.plot_confusion_matrix(test_labels, preds, ['Baseline', 'Endpoint'])
        wandb.sklearn.plot_roc(test_labels, probs, ['Baseline', 'Endpoint'])
        wandb.sklearn.plot_precision_recall(test_labels, probs, ['Baseline', 'Endpoint'])


def score_model(labels: np.ndarray, predictions: np.ndarray):
        '''
        generate all metrics for a given set of labels and predictions
        '''

        metrics = {'accuracy':accuracy_score(labels, predictions),
                   'balanced_accuracy':balanced_accuracy_score(labels, predictions),
                   'precision':precision_score(labels, predictions),
                   'recall':recall_score(labels, predictions),
                   'f1':f1_score(labels, predictions)
        }
        return metrics 
