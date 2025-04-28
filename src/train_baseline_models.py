
import pandas as pd
from train_deployment_models import load_test_indices, get_diet_indices, create_timepoint_labels, load_diet_labels, cv_search, logging



def load_data(time_index, diet_index):
    '''
    load raw rank normalized data
    '''
    fp = 'data/raw/microbe_metabolites_filtered_rank_normalized.csv'
    data = pd.read_csv(fp, index_col='sample').fillna(0)
    return data.loc[time_index], data.loc[diet_index]

def main():
    '''
    main function to train models to evaluate embedding space using CV
    and extract final weights from full model 
    '''
    # get indices for each split 
    time_indices = load_test_indices()
    diet_indices = get_diet_indices(time_indices)
    # load labels 
    time_labels = create_timepoint_labels(time_indices['nodes'])
    diet_labels = load_diet_labels(diet_indices['nodes'])
    # load embedding 
    time_data, diet_data = load_data(time_indices['nodes'], diet_indices['nodes'])
    # define model and parameter space 
    time_search = cv_search(time_indices, time_data, time_labels)
    diet_search = cv_search(diet_indices, diet_data, diet_labels)
    logging(time_search=time_search, 
            diet_search=diet_search, 
            file_path='results/best/baseline_logging.txt')

if __name__ == '__main__':
    main()