'''
Author: Keenan Manpearl
Date: 2024-09-09

This script trains a node classifier to predict 
baseline vs endpoint samples.

'''

import wandb
import pickle
import numpy as np 
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from data import MultiomicsEmbedding


NUM_CV_FOLDS = 10
MAX_ITER = 1000


def log_cv(search, classifier_type: str, model_num: int):
    '''
    parse and log the results of CV search to wandb
    '''
    results = search.cv_results_
    best_idx = search.best_index_
    scores = {i: results[f'split{i}_test_score'][best_idx] for i in range(NUM_CV_FOLDS)}
    for fold, score in scores.items():
        wandb.summary[f'{classifier_type}_model_{model_num}_val_{fold}_f1'] = score


def cv(data: dict, classifier_type: str, model_num: int, seed: int) -> tuple[dict, int]:
    '''
    use a RandomGridSearch to find the best hyperparameters
    '''
    train_data = data['train_data']
    train_labels = data['train_labels']
    params =[ #{'C': uniform(0.1, 500.0),
                #'penalty': ['l1'],
                #'solver': ['liblinear']},
                #{'C': uniform(0.1, 500.0),
                #'penalty': ['l2'],
                #'solver': ['lbfgs']},
                {'C': uniform(0.1, 500.0),
                'penalty': ['elasticnet'],
                'l1_ratio': uniform(0, 1),
                'solver': ['saga']
                }]
    log_reg = LogisticRegression(random_state=seed, max_iter=MAX_ITER)
    clf = RandomizedSearchCV(log_reg, params, n_iter=500, cv=NUM_CV_FOLDS, scoring='f1', random_state=seed)
    search = clf.fit(train_data, train_labels)
    log_cv(search, classifier_type, model_num)
    return search.best_params_, search.best_score_

def train(data:dict, params:dict, seed: int) -> LogisticRegression :
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


def train_loop(p: float, q: float, gamma: int, random_seed: int, n2v_mode: str, save: bool):
    '''
    trains and evaluates all models associated with an embedding space
    '''
    scores = {'time': [], 'diet': []}
    for model in range(1,6):
        # load data assocaited with model 
        data = MultiomicsEmbedding(model, p, q, gamma, random_seed, n2v_mode)
        for classifier_type in ['time', 'diet']:
            dataset = data.datasets[classifier_type]
            best_params, avg_score = cv(dataset, classifier_type, model, random_seed)
            # record parameters found through cv
            log_model(classifier_type, model, best_params)
            # record avg score for CV
            scores[classifier_type].append(avg_score)
            # train final classifer using all training data 
            clf = train(dataset, best_params, random_seed)
            eval(clf, dataset['train_data'], dataset['train_labels'], 'train', classifier_type, model)
            eval(clf, dataset['test_data'], dataset['test_labels'], 'test', classifier_type, model)
            if save:
                with open(f'results/models/{classifier_type}_{model}_p_{p}_q_{q}_g_{gamma}.pkl', 'wb') as f:
                    pickle.dump(clf, f)
    log_final_metrics(scores)