# Self-organization of spermatogenic wave coordinates sustained sperm production in the mouse testis

This repository contains all the code used in the paper, including: 
- Simulation and inference of the Sakaguchi-Kuramoto(SK) model and stochastic Sakaguchi-Kuramoto model in 1D.
- Theory for persistent oscillations of retinoic acid concentration.

To install the current packages, you can either use `conda` to setup a local python environment using `environment.yml` or install all the dependencies listed in `environment.yml` manually. 

## Notebooks used to plot figures and perform analysis in the ST
1. `kuramoto1D_sims.ipynb`: simulations of 1D SK model with various parameters, including the inferred MAP parameters.
2. `kuramoto2D_sims.ipynb`: simulations of 2D SK model in a narrow tube. 
3. `inference_from_data.ipynb`: inference of MAP parameters from the longest segments of 9 tubules.
4. `inference_validation_eta.ipynb` and `inference_validation_sigma.ipynb`: validation of the inference scheme on simulated datasets with varying $\eta$ and $\sigma$.
5. `kuramotoVF_sims.ipynb`: simulations of 1D stochastic SK model. 
6. `matching_with_data.ipynb`å: matching stochastic SK model with data by comparing square deviations from the auto-correlation of theta in data. 
7. `RA_model_0D.ipynb`: simulations of the RA model, including exploring its phase diagram. This is without the spatial extent. 
8. `RA_model_1D.ipynb`: simulations of the spatially coupled RA model.
9. `plot_potential.ipynb`: code for plotting the effective potential for the fully synchronised SK model.
10. `lin_stability.ipynb`: analytical linear stability analysis of the RA model.

## Exploratory notebooks, not directly used in figure plotting but aids the analysis in the ST
1. `inference_from_data_using_shortloops.ipynb`: inference using short loops of the 9 tubules. The outcome is the same as using the longest segments, so the results are not included in the final manuscript.
2. `network.ipynb`: exploring the effect of non-nearest-neighbour couplings. 
3. `kymograph.ipynb`: plotting from .tif files from live-image data.
4. `inference_from_data_NNN.ipynb`: inference using the next-nearest-neighbour SK model. 
5. `inference_validation_NNN.ipynb`: validation of the inference scheme in the next-nearest-neighbour SK model. 
6. `kuramoto_quasi1D_sims.ipynb`: simulations of the quasi-1D SK model. 
7. `kuramotoNNN_sims.ipynb`: simulations of the next-nearest-neighbour SK model. 

## Functions used by the notebooks 
1. `kuramoto.py`: numerical simulator for the SK model in 1D, 2D and as a network.
2. `lin_stability.py`: implementation of several root finding algorithms in the linear stability analysis.
3. `sto_density.py`: numerical simulator for the RA concentration model in 0D and 1D.
4. `utils.py`: utility functions mainly used in data processing. 




