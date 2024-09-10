'''
Author: Keenan Manpearl
Date: 2024-09-09

This script trains a node classifier to predict 
baseline vs endpoint samples.

'''

import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


class Trainer:
    def __init__(self, penalty, c):
        solver = self.get_solver()
        self.model = LogisticRegression(penalty=penalty, C=c, solver=solver)
        self.metrics = {'train_accuracy': {}, 
                        'val_accuracy': {}, 
                        'train_balanced_accuracy': {}, 
                        'val_balanced_accuracy': {}, 
                        'train_precision': {}, 
                        'val_precision': {}, 
                        'train_recall': {}, 
                        'val_recall': {}, 
                        'train_f1': {}, 
                        'val_f1': {}
            }

    def get_solver(penalty):
        if penalty == 'l1':
            return 'liblinear'
        else:
            return 'lbfgs'
        
    def score_model(self, split, cv_fold, labels, predictions):
        self.metrics[f'{split}_accuracy'][cv_fold] = accuracy = accuracy_score(labels, predictions)
        self.metrics[f'{split}_balanced_accuracy'][cv_fold] = balanced_accuracy = balanced_accuracy_score(labels, predictions)
        self.metrics[f'{split}_precision'][cv_fold] = precision = precision_score(labels, predictions)
        self.metrics[f'{split}_recall'][cv_fold] = recall = recall_score(labels, predictions)
        self.metrics[f'{split}_f1'][cv_fold] = f1 =f1_score(labels, predictions)

        wandb.summary[f'{split}_accuracy_{cv_fold}'] = accuracy
        wandb.summary[f'{split}_balanced_accuracy_{cv_fold}'] = balanced_accuracy
        wandb.summary[f'{split}_precision_{cv_fold}'] = precision
        wandb.summary[f'{split}_recall_{cv_fold}'] = recall
        wandb.summary[f'{split}_f1_{cv_fold}'] = f1
    
    def evaluate(self, val_data, val_labels, train_data, train_labels, cv_fold):
        train_preds = self.model.predict(train_data)
        val_preds = self.model.predict(val_data)
        val_probs = self.model.predict_proba(val_data)
        self.score_model('train', cv_fold, train_labels, train_preds)
        self.score_model('val', cv_fold, val_labels, val_preds)
        wandb.sklearn.plot_summary_metrics(self.model, train_data, train_labels, val_data, val_labels)
        wandb.sklearn.plot_roc(val_labels, val_probs, ['Baseline', 'Endpoint'])
        wandb.sklearn.plot_precision_recall(val_labels, val_probs, ['Baseline', 'Endpoint'])
    
    def log_scores(self):
        for metric in self.metrics.keys():
            avg = 0
            for score in self.metrics[metric].values():
                avg += score
            self.metrics[metric]['avg'] = avg_score = avg / 5
            
            wandb.summary[f'{metric}_avg'] = avg_score
            wandb.log({f'{metric}_avg': avg_score})

    def train(self, data):
        for cv_fold in [0,1,2,3,4]:
            dataset = data.datasets[cv_fold]
            train_data = dataset['train_data']
            train_labels = dataset['train_labels']
            val_data = dataset['val_data']
            val_labels = dataset['val_labels']
            self.model.fit(train_data, train_labels)
            self.evaluate(val_data, val_labels, train_data, train_labels, cv_fold)     
        self.log_scores()
