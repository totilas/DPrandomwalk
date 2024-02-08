# Differentially Private Decentralized Learning with Random Walks

Here is the code used to generate figures. 

## Requirements
numpy
networkx
matplotlib
tqdm
scipy

for the houses part:
sklearn
Pathlib
typer

## Datasets

The Facebook ego graph can be downloaded here: https://snap.stanford.edu/data/ego-Facebook.html and should be placed in a folder named 'facebook' in the project folder. The Housing dataset should be downloaded via the 'data' script, but can be manually retrieved there in case of problem: https://www.openml.org/d/823


# Organization

- The file `synthemuffcomparison.py` enables to reproduce figure 1
- The folder Houses enable to reproduce the experiments of Figure 2
- The files `south.py` and `fb.py` enable to reproduce Figure 3


The code is adapted from the code of [Muffliato: Peer-to-Peer Privacy Amplification for Decentralized Optimization and Averaging.](https://github.com/totilas/muffliato).