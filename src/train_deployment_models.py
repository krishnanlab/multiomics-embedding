
'''
Author: Keenan Manpearl
Date: 2024-09-09

This script trains a classifier using all availble samples per fold.
Training/evaluations are done using samples (infants)
and predictions are made for features (microbes and metabolites).

'''


import pandas as pd
import numpy as np
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from argparse import ArgumentParser
from datetime import datetime
import warnings
import pickle
import os
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


SEED = 22
MAX_ITER = 100
N_MODELS = 500


def load_test_indices():
    '''
    Load the test indices for cross validation
    '''
    labels = pd.read_csv('data/raw/sample_breakdown.csv')
    subset = labels[labels['partition'] == 'test']
    return subset[['nodes', 'run']]

def get_diet_indices(time_indices):
    '''
    Get the endpoint samples for diet classification 
    '''
    return time_indices.loc[time_indices['nodes'].str.contains("End")]


def load_embedding(p, q, g, time_index, diet_index):    
    '''
    load embedding file and return sample embeddings 
    '''
    emb_file = f'data/best_emb/emb_p_{p}_q_{q}_g_{g}.tsv.gz'
    emb = pd.read_csv(emb_file, sep='\t', index_col=0)
    return emb.loc[time_index], emb.loc[diet_index]


def create_timepoint_labels(time_index):
    '''
    Convert time points into binary labels for ML
    '''
    return np.array([1 if "End" in item else 0 for item in time_index])

def load_diet_labels(diet_index):
    '''
    Convert diet into binary labels for ML
    '''
    labels = pd.read_csv('data/raw/microbiome_info_data.csv', usecols=['sample.name', 'Group', 'Time'])
    labels.index = labels['sample.name']
    labels['Label'] = labels['Group'].map({'Dairy': 0, 'Meat': 1})
    labels = labels[~labels.index.duplicated(keep='first')]
    return labels.loc[diet_index, ['Label']].to_numpy().ravel()

def cv_search(split_indices, emb, labels):
    '''
    perform a randomized search to find the best hyperparameters
    '''
    log_reg = LogisticRegression(random_state=SEED, max_iter=MAX_ITER)
    params =[ {'C': uniform(0.1, 500.0),
                'penalty': ['l1'],
                'solver': ['liblinear']},
                {'C': uniform(0.1, 500.0),
                'penalty': ['l2'],
                'solver': ['lbfgs']},
                {'C': uniform(0.1, 500.0),
                'penalty': ['elasticnet'],
                'l1_ratio': uniform(0, 1),
                'solver': ['saga']
                }]
    custom_cv = PredefinedSplit(split_indices['run'])
    clf = RandomizedSearchCV(estimator=log_reg, 
                             param_distributions=params, 
                             n_iter=N_MODELS, 
                             cv=custom_cv, 
                             scoring=['f1', 'balanced_accuracy', 'accuracy'], 
                             refit='f1',
                             random_state=SEED)
    return clf.fit(emb, labels)


def get_scores(results, best_idx):
    '''
    extract the scores from the cross validation results
    '''
    scores = {}
    for metric in ['f1', 'balanced_accuracy', 'accuracy']:
        scores[metric] = {i: results[f'split{i}_test_{metric}'][best_idx] for i in range(5)}
    return scores

def write_params(param_dict, file):
    '''
    write the best parameters to a file
    '''
    for param, value in param_dict.items():
        file.write(f'best {param}: {value}\n')

def calculate_iqr(scores):
    '''
    calculate the interquartile range of the scores
    '''
    return np.percentile(scores, 75) - np.percentile(scores, 25)

def write_scores(search, file):
    '''
    write the scores to a file
    '''
    score_dict = get_scores(results=search.cv_results_, best_idx=search.best_index_)
    for metric, scores in score_dict.items():
        file.write(f'{metric} validation scores\n')
        for fold, s in scores.items():
            file.write(f'fold {fold}: {s}\n')
        file.write(f'median score: {np.median(list(scores.values()))}\n')
        file.write(f'IQR: {calculate_iqr(list(scores.values()))}\n')
        file.write(f'variance: {np.var(list(scores.values()))}\n')
        file.write('\n')
    

def logging(time_search, diet_search, file_path, p=None, q=None, g=None):
    '''
    log all model results to a file
    '''
    with open(file_path, 'w') as f:
        if p is not None:
            f.write('============== Node2Vec Parameters ==============\n') 
            f.write(f'p: {p}\n')
            f.write(f'q: {q}\n')
            f.write(f'gamma: {g}\n')
            f.write('\n')
        f.write('============== Timepoint Classification ==============\n') 
        write_params(param_dict=time_search.best_params_, file=f)
        f.write('\n')
        write_scores(search=time_search, file=f)
        f.write('\n')
        f.write('============== Diet Classification ==============\n') 
        write_params(param_dict=diet_search.best_params_, file=f)
        f.write('\n')
        write_scores(search=diet_search, file=f)

def save_model_weights(model, model_type, out_root, tag):
    '''
    save the model weights to a file
    '''
    coefficients = model.coef_[0]
    with open(f'{out_root}/{tag}_{model_type}_model_weights.txt', 'w') as f:
        f.writelines(f'{coef}\n' for coef in coefficients)

def save_model(model, model_type, out_root, tag):
    '''
    save the model to a file
    '''
    with open(f'{out_root}/{tag}_{model_type}_model.pkl', 'wb') as f:
        pickle.dump(model, f)

def generate_feature_predictions(model,model_type,out_root,tag,p,q,g):
    '''
    generate predictions for each feature in the embedding
    and save to file 
    '''
    colnames ={'time': ['baseline', 'endpoint'], 'diet': ['dairy', 'meat']}
    # load all embeddings
    emb_file = f'data/best_emb/emb_p_{p}_q_{q}_g_{g}.tsv.gz'
    emb = pd.read_csv(emb_file, sep='\t', index_col=0)
    # get features 
    features = emb[~emb.index.str.startswith('MD')]
    # feature predictions
    preds = model.predict_proba(features)
    preds = pd.DataFrame(preds, index=features.index, columns=colnames[model_type])
    preds.to_csv(f'{out_root}/{tag}_{model_type}_feature_predictions.tsv', sep='\t')


def train_full_model(best_params, emb, labels, model_type, out_root, tag, p ,q ,g):
    '''
    train the full model with the best hyperparameters
    save model, weights, and feature predictions
    '''
    log_reg = LogisticRegression(random_state=SEED, max_iter=MAX_ITER)
    log_reg.set_params(**best_params)
    log_reg.fit(emb, labels)
    save_model_weights(model=log_reg, 
                       model_type=model_type,
                       out_root=out_root,  
                       tag=tag)
    save_model(model=log_reg, 
               model_type=model_type, 
               out_root=out_root, 
               tag=tag)
    generate_feature_predictions(model=log_reg,
                                 model_type=model_type,
                                 out_root=out_root, 
                                 tag=tag,
                                 p=p,
                                 q=q,
                                 g=g)



def setup_output_dir(out_dir):
    '''
    create output directory if it does not exist
    '''
    if out_dir is None:
        current_date = datetime.now().strftime('%Y-%m-%d')
        out_dir = f'results/best_{current_date}'
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def main(p,q,g, out_dir, tag):
    '''
    main function to train models to evaluate embedding space using CV
    and extract final weights from full model 
    '''

    print(tag)
    # get indices for each split 
    time_indices = load_test_indices()
    diet_indices = get_diet_indices(time_indices)
    # load labels 
    time_labels = create_timepoint_labels(time_indices['nodes'])
    diet_labels = load_diet_labels(diet_indices['nodes'])
    # load embedding 
    time_emb, diet_emb = load_embedding(p,q,g, time_indices['nodes'],diet_indices['nodes'])
    # define model and parameter space 
    time_search = cv_search(time_indices, time_emb, time_labels)
    diet_search = cv_search(diet_indices, diet_emb, diet_labels)
    logging(time_search=time_search, 
            diet_search=diet_search, 
            file_path=f'{out_dir}/{tag}_logging.txt', 
            p=p, 
            q=q, 
            g=g)
    train_full_model(best_params=time_search.best_params_, 
                     emb=time_emb, 
                     labels=time_labels, 
                     model_type='time', 
                     out_root=out_dir,
                     tag=tag,
                     p=p,
                     q=q,
                     g=g)
    train_full_model(best_params=diet_search.best_params_, 
                     emb=diet_emb, 
                     labels=diet_labels, 
                     model_type='diet', 
                     out_root=out_dir,
                     tag=tag,
                     p=p,
                     q=q,
                     g=g)
    
    
   

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--out",
                        help="output directory to save files",
                        required=False,
                        type=str,
                        default=None)
    args = parser.parse_args()
    out_dir = setup_output_dir(args.out)

    models = {'wcksnlsg' : {'p' : 19.0, 'q' : 9.122152261131532, 'g' : 1},
              'ai9n4jxs' : {'p' : 0.8055551041134607, 'q' : 0.1, 'g' : 1},
              '7o4yga2v' : {'p' : 0.5, 'q' : 1.895944090041435, 'g' : 1},
              '21tdsqsa' : {'p' : 1.0795506927238254, 'q' : 8.383911078685804, 'g' : 1},
              'q2gzu1o3' : {'p' : 19.0, 'q' : 8.483911078685804, 'g' : 2},
              '8lofhbbf' : {'p' : 7.305688086564288, 'q' : 7.517332462471247, 'g' : 2},
              'qb4y98x0': {'p': 5.5, 'q': 9.010757520712524, 'g': 1}}
    
    for model, params in models.items():
        main(p=params['p'], q=params['q'], g=params['g'], out_dir=out_dir, tag=model)
