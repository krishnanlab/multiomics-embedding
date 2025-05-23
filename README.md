This repository contains code and data to accompany the paper **Network-based representation learning reveals the impact of age and diet on the gut microbial and metabolomic environment of U.S. infants in a randomized controlled feeding trial** [doi.org/10.1101/2024.11.01.621627](https://www.biorxiv.org/content/10.1101/2024.11.01.621627v1). This includes preprocessing the original microbial and metabolomic count data, creating a sample X feature edge list where the edge weight between two nodes is their normalized count value, creating node2vec+ embeddings, selecting embedding spaces, and using embeddings to train diet and time point classifiers. 


## Installation

All python package dependencies may be installed using conda. 
If you do not already have conda installed see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) for installation instruction. 

Then run the following:

```
git clone git@github.com:krishnanlab/multiomics-embedding.git
cd multiomics-embedding
conda env create -f environment.yml
```


## Usage

For ease of use, `run/` contains shell scripts that call the python code in `src/`. 
All run scripts should be invoked from the project root.
Each script’s header includes its required arguments and flags.
Run `bash scripts/<name>.sh --help` for details.

`run_initial_sweep.sh` was used to evaluate node2vec+ embedding parameters effect on time point classifiers.

`run_joint_sweep.sh` was used to evaluate embedding parameters on both time point and diet classifiers. 

`run_all.sh` was used to compare all unique embedding spaces generated during the two sweeps.

`run_baseline.sh` was used to train logistic regression models using the processed -omics counts directly as features.

`run_deployment.sh` was used to train logistic regression models using embedding features and find -omics features that are predicted to be associted with a diet or time point phenotype. 

## Repository Organization 

```
├── data/            # raw and processed data
├── notebooks/       # exploratory analysis 
├── src/             # main code
├── results/         # all results for top performing embedding spaces
├── run/             # shell scripts to call run code
├── environment.yml  # conda environment

```
In this repository we only include data and results for our top performing embedding spaces which were used in the paper. The performance of other embedding spaces can be seen in our [public wandb project](https://wandb.ai/keenan-manpearl/multiomics_embedding). Variation of all models is explored in [src/2024-12-13_model_variance.ipynb](https://github.com/krishnanlab/multiomics-embedding/blob/main/src/2024-12-13_model_variance.ipynb)


## License 
This repository and all its contents are released under the BSD 3-Clause License[https://opensource.org/license/BSD-3-Clause]; See LICENSE[https://github.com/krishnanlab/multiomics-embedding/blob/main/LICENSE]


## Original Data

Link to paper from original study?

## Citation 

## Authors 

## Funding

