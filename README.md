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
Each script’s header comment documents its usage and any required arguments.

The pipeline runs in this order:

1. `run_initial_sweep.sh <wandb_username>` — sweep node2vec+ parameters (p, q, gamma) and evaluate their effect on time point classifiers only.
2. `run_joint_sweep.sh <wandb_username>` — sweep node2vec+ parameters and evaluate their effect on both time point and diet classifiers jointly.
3. `run_all.sh` — compare all unique embedding spaces generated during the two sweeps and select the top performers.
4. `run_baseline.sh` — train logistic regression models using the processed -omics counts directly as features, as a baseline for comparison against embedding-based models.
5. `run_deployment.sh` — train logistic regression models using the selected embedding features and identify -omics features predicted to be associated with a diet or time point phenotype.

## Data

`data/raw/` contains the microbiome and metabolomics abundance data underlying this project, plus sample metadata and CV splits; see [data/README.md](data/README.md) for a full description of each file. Briefly:

- Microbial features are gene functional annotations (KEGG `K#####` orthologs, `COG####` clusters, eggNOG `ENOG#...` groups) and taxonomic lineage strings (`k_...p_...c_...o_...f_...g_...s_...`).
- Metabolite features are anonymized compound IDs split by extraction method (`N_AQ.###` for aqueous, `N_LP.###` for lipid).

## Repository Organization 

```
├── data/            # raw and processed data
├── notebooks/       # exploratory analysis 
├── src/             # main code
├── results/         # all results for top performing embedding spaces
├── run/             # shell scripts to call run code
├── environment.yml  # conda environment

```
In this repository we only include data and results for our top performing embedding spaces which were used in the paper. The performance of other embedding spaces can be seen in our [public wandb project](https://wandb.ai/keenan-manpearl/multiomics_embedding). Variation of all models is explored in [notebooks/2024-12-13_model_variance.ipynb](https://github.com/krishnanlab/multiomics-embedding/blob/main/notebooks/2024-12-13_model_variance.ipynb)


## License 
This repository and all its contents are released under the [BSD 3-Clause License](https://opensource.org/license/BSD-3-Clause); See [LICENSE](https://github.com/krishnanlab/multiomics-embedding/blob/main/LICENSE)

## Authors 
Adelle Price, Sakaiza Rasolofomanana-Rajery, Keenan Manpearl, Charles E. Robertson, Nancy F. Krebs, Daniel N. Frank, Arjun Krishnan*, Audrey E. Hendricks*, Minghua Tang*  
*These authors contributed equally.

## Funding
NIH (NIDDK) 1K01DK111665-01, 1R01DK126710, the Beef Checkoff through the National Cattlemen’s Beef Association, and the National Pork Board.

## Citation
The paper associated with this codebank is:

> Price A, Rasolofomanana-Rajery S, Manpearl K, Robertson CE, Krebs NF, Frank DN, Krishnan A, Hendricks AE, Tang M. Network-based representation learning reveals the impact of age and diet on the gut microbial and metabolomic environment of U.S. infants in a randomized controlled feeding trial. *bioRxiv*. 2024. doi: [10.1101/2024.11.01.621627](https://www.biorxiv.org/content/10.1101/2024.11.01.621627v1)

```bibtex
@article{price2024network,
  title={Network-based representation learning reveals the impact of age and diet on the gut microbial and metabolomic environment of U.S. infants in a randomized controlled feeding trial},
  author={Price, Adelle and Rasolofomanana-Rajery, Sakaiza and Manpearl, Keenan and Robertson, Charles E and Krebs, Nancy F and Frank, Daniel N and Krishnan, Arjun and Hendricks, Audrey E and Tang, Minghua},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.11.01.621627}
}
```
