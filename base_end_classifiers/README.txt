File definitions in the 'base_end_classifiers' folder

full_data_pecan_3.tsv : microbiome and metabolome data filtered for missingness <90% and (missingness >10% or cv>2nd%tile), data is rank normalized and in graph format ready for input into pecanpy

Generate_Pecan_Network.ipynb : contains code to perform graph embedding using full_data_pecan_3.tsv as the data input

data_128_dim.txt : output from Generate_Pecan_Network.ipynb

microbiome_info_data.csv : contains data labels and sample IDs for classification model building

Build_Base_End_Classifiers.R : code for formatting 'data_128_dim.txt' and building and assessing base/end classifiers