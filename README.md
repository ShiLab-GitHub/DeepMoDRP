# DeepMoDRP
This repository archives the datasets and Python scripts employed in the study described in the paper ["DeepMoDRP: A Multi‐Omics‐Based Deep Learning Framework for Drug Response Prediction in Brain Cancer"](https://doi.org/10.1002/minf.70020). The published article is freely available via [the link](https://onlinelibrary.wiley.com/share/author/NSQUGAQNMVVR7BYSJE6P?target=10.1002/minf.70020).
## Data
1. ic50/80cell_line_ic50.csv: brain cancer drug response data from the GDSC2 database  
2. cellline/cellline_listwithACH_80cellline.csv: list of cell lines we used  
3. cellline/80cellline_17737dim_RNAseq.csv: feature of gene expression 
4. cellline/CNV_85dim_23430dim.csv: feature of copy number variation
5. cellline/METH_84cellline_378dim.csv: feature of methlylation
6. cellline/MUT_85dim_17514dim.csv: feature of mutation
7. cellline/MUT_85dim_2028dim.csv: feature of mutation delete 0
8. cellline/512dim_RNAseq.pkl: pre-processed gene expression data used DAE and SAE
9. cellline/512dim_copynumber.pkl: pre-processed copy number variation data used DAE and SAE
10. drug/smile_inchi.csv: list of drugs and their SMILES (Simplified molecular input line entry system)
## Source codes
1. preprocess.py: load data and convert to pytorch format  
2. train.py: train the model and make predictions  
3. functions.py: some custom functions  
4. simles2graph.py: convert SMILES sequence to graph  
5. SAEandDAE_CNV.py: dimensionality reduction for copy number variation
6. SAEandDAE_RNAseq.py: dimensionality reduction for gene expression
6. MofaDRP_Net.py: multi-omics fusion autoencoder network model 
## Step-by-step instructions
1. Install dependencies, including torch1.4, torch_geometric (you need to install torch_cluster, torch_scatter, torch_sparse, and torch_spline_conv before installation), matplotlib, scipy, sklearn, rdkit, and networkx.  
2. Run SAEandDAE_CNV.py to reduce the dimensionality of the copynumber feature.
3. Run SAEandDAE_RNAseq.py to reduce the dimensionality of the gene expression feature.
4. Run preprocess.py to convert label data and feature data into pytorch format.
5. Run train.py for training and prediction.
## Dependencies 
python == 3.7.10
pytorch = 1.12.1
torch_geometric = 2.3.0(install torch_cluster = 1.6.0, torch_scatter = 2.1.0, torch_sparse = 0.6.15, torch_spline_conv = 1.2.1 before installation) https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html  
matplotlib = 3.5.3
scipy = 1.7.3
sklearn = 0.24.2
rdkit = 2023.3.2
networkx = 2.6.3
