import networkx as nx
import pandas as pd
import community as community_louvain
from collections import defaultdict
from itertools import combinations
import numpy as np
import argparse
import matplotlib.pyplot as plt

#Command line arguments
parser = argparse.ArgumentParser(description="Run Louvain Clustering multiple times and extract connected components.")
parser.add_argument("--NumLouvain", type=int, default=100, help="Number of times to run Louvain clustering.")
parser.add_argument("--Cooccur_percent", type=float, default=0.9, help="Percentage threshold for co-occurrence.")
parser.add_argument("--Resolution", type=float, default=1.0, help="Resolution parameter for Louvain clustering.")
args = parser.parse_args()
#Assigning command line arguments
k = args.NumLouvain
percentkeep = args.Cooccur_percent
resolution = args.Resolution 

#Creating a fake graph for testing purposes
sizes = [50, 50, 50, 50, 50]
p_intra = 0.8 
p_inter = 0.02  

# Create a stochastic block model graph
prob_matrix = [[p_intra if i == j else p_inter for j in range(len(sizes))] for i in range(len(sizes))]
G = nx.stochastic_block_model(sizes, prob_matrix, seed=7)
#Graphing my fake graph to make sure things make sense
#nx.draw(G, with_labels=False, node_size=10)
#plt.show()

#Louvain clustering K times and storing results in a dictionary
partitions = []
nodes = list(G.nodes)
node_idx = {node: idx for idx, node in enumerate(nodes)}  # Mapping nodes to indices
num_nodes = len(nodes)

#Where cooccurance counts are stored for the louvain clustering loop
co_occurrence_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

#Loop where louvain clustering is done
for _ in range(k):
    partition = community_louvain.best_partition(G, resolution=resolution)

    #Going through each found louvain cluster and updating the cooccurance counts
    for community in set(partition.values()):
        community_nodes = [node for node, comm in partition.items() if comm == community]

        #Update co-occurrence matrix
        for i, j in combinations(community_nodes, 2):
            co_occurrence_matrix[node_idx[i], node_idx[j]] += 1

#Creating a data frame out of the matrix
co_occurrence_df = pd.DataFrame(
    [
        (nodes[i], nodes[j], co_occurrence_matrix[i, j])
        for i in range(num_nodes) for j in range(i + 1, num_nodes)
    ],
    columns=["Source", "Target", "CoOccurrence"])

#Filtering edges based on threshold to find highly cooccuring gene pairs
threshold = percentkeep * k
filtered_edges = co_occurrence_df[co_occurrence_df["CoOccurrence"] >= threshold]

#Creating a new graph out of the highly cooccured edges
G_new = nx.Graph()
G_new.add_edges_from(zip(filtered_edges['Source'], filtered_edges['Target']))

#Getting connected components of this new graph
components = list(nx.connected_components(G_new))

#Creating dataframe where listing the Cluster, and Gene in 2 columns
rows = [{"CC": f"Component_{i + 1}", "Gene": gene} for i, component in enumerate(components) for gene in component]
cc_df = pd.DataFrame(rows)

#Save the final clusters (connected components from graph made of occuring louvain cluster assignments)
cc_df.to_csv("connected_components.csv", index=False)
print("Connected components written to 'connected_components.csv'.")

#Drawing reconciled communities to make sure things make sense
#nx.draw(G_new, with_labels=False, node_size=10)
#plt.show()