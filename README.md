This project is a collaboration with Adelle Price from the Hendricks lab.
The goal is to find metabolites and microbes that are associated
with diet intervention phenotypes. 

To do this we will first create a heterogenous network.
Samples (Baseline and Endpoint per individual), 
metbolites and microbes will all be represented as nodes.
Edges will represent rank normalized abundance of 
microbes and metabolites in samples.
Then we will train two node classifiers on samples:
Baseline vs endpoint and meat vs dairy diet intervention.
Once trained, we will use the classifier to predict
microbes and metabolites that are also assoicated with a 
specific phenotype (baseline vs endpoint or diet intervention). 

