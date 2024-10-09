'''
Author: Keenan Manpearl
Date: 2024-09-09

This script trains a node classifier to predict 
baseline vs endpoint samples.

'''

import wandb
import numpy as np 
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


NUM_CV_FOLDS = 10


def log_cv(search, classifier_type: str, model_num: int):
    '''
    parse and log the results of CV search to wandb
    '''
    results = search.cv_results_
    best_idx = search.best_index_
    scores = {i: results[f'split{i}_test_score'][best_idx] for i in range(NUM_CV_FOLDS)}
    scores['avg'] = search.best_score_
    for fold, score in scores.items():
        wandb.summary[f'{classifier_type}_model_{model_num}_val_{fold}_f1'] = score

def cv(data: dict, classifier_type: str, model_num: int, seed: int):
    '''
    use a RandomGridSearch to find the best hyperparameters
    '''
    train_data = data['train_data']
    train_labels = data['train_labels']
    params = [{'C': uniform(0.1, 500.0),
                'penalty': ['l2'],
                'solver': ['liblinear']
                },
                {'C': uniform(0.1, 500.0),
                'penalty': ['l1'],
                'solver': ['liblinear']
                },
                {'C': uniform(0.1, 500.0),
                'penalty': ['elasticnet'],
                'l1_ratio': uniform(0.01, .99),
                'solver': ['saga']
                }]
    log_reg = LogisticRegression(random_state=seed, max_iter=500)
    clf = RandomizedSearchCV(log_reg, params, n_iter=500, cv=NUM_CV_FOLDS, scoring='f1', random_state=seed)
    search = clf.fit(train_data, train_labels)
    log_cv(search, classifier_type, model_num)
    return search.best_params_, search.best_score_


def train(data:dict, params:dict, seed: int):
    '''
    train a logistic regression model on the training data
    '''
    train_data = data['train_data']
    train_labels = data['train_labels']
    if params['penalty'] == 'elasticnet':
        model = LogisticRegression(C=params['C'], 
                                   penalty=params['penalty'], 
                                   solver=params['solver'], 
                                   l1_ratio=params['l1_ratio'], 
                                   random_state=seed, 
                                   max_iter=500)
    else:
        model = LogisticRegression(C=params['C'], 
                                   penalty=params['penalty'], 
                                   solver=params['solver'], 
                                   random_state=seed, 
                                   max_iter=500)
    model.fit(train_data, train_labels)
    return model

        
def eval(classifier: LogisticRegression, data: pd.DataFrame, labels: np.ndarray, split: str, classifier_type: str, model_num: int):
    '''
    generate and log all metrics for a given split (train or test)
    '''

    predictions = classifier.predict(data)
    scores = {'accuracy':accuracy_score(labels, predictions),
                'balanced_accuracy':balanced_accuracy_score(labels, predictions),
                'precision':precision_score(labels, predictions),
                'recall':recall_score(labels, predictions),
                'f1':f1_score(labels, predictions)}
    for metric, score in scores.items():
        wandb.log({f'{classifier_type}_model_{model_num}_{split}_{metric}': score})
        wandb.summary[f'{classifier_type}_model_{model_num}_{split}_{metric}'] = score

def log_model(classifier_type:str, model:int, params:dict):
    '''
    log model hyperparameters to wandb
    '''
    for parameter, value in params.items():
        wandb.summary[f'{classifier_type}_model_{model}_{parameter}'] = value

def log_final_metrics(scores: dict):
    '''
    log aggregate metrics for each classifier 
    and an overall embedding score to wandb 
    '''
    emb_score = 0
    for classifier_type, avg_scores in scores.items():
        classifier_score = sum(avg_scores)/len(avg_scores)
        wandb.summary[f'{classifier_type}_avg_val_f1'] = classifier_score
        emb_score += classifier_score
    wandb.summary['emb_score'] = emb_score/2

