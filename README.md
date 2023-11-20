# Huang2023_ModelVariabilityWithSubjectEmbeddings
This repository consists of scripts for reproducing results in the "Modelling subject variability in dynamic functional
brain networks using embeddings" manuscript.

## Requirements
- Scripts for preprocessing data depends on the [osl](https://github.com/OHBA-analysis/osl) toolbox. 
- Scripts for training models and analysing results depends on the [osl-dynamics](https://osl-dynamics.readthedocs.io) toolbox, which includes source code for the SE-HMM, SE-DyNeMo models, as well as analysis tools.

## Contents
```data_preprocessing```: This directory contains scripts for preprocessing, coregistration, source reconstruction and fixing sign ambiguity for the three MEG datasets used.

```simulations```: This directory contains scripts for simulation analysis on SE-HMM.
- ```1_multivariate.py```: This script shows how multivariate information is captured in SE-HMM.
- ```2_recover_structure.py```: This script shows how the underlying population structure is recovered by SE-HMM.
- ```3_acc_increase_with_nsubjects.py```: This script shows SE-HMM performs more accurate inference than HMM with dual-estimation and can make use of increasing amount of data with different characteristics.

```real_data```: This directory contains scripts for training SE-HMM and perform analysis on three MEG datasets.
- ```wakeman_henson```: This directory contains scripts for training SE-HMM on the Wakeman-Henson dataset.
- ```multi_dataset```: This directory contains scripts for training SE-HMM on combined resting-state data from the MRC MEGUK nottingham site dataset and the Cam-CAN dataset.
- ```camcan```: This directory contains scripts for training SE-HMM on the Cam-CAN dataset resting-state data.
