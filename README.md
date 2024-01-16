# Huang2023_ModelVariabilityWithSubjectEmbeddings
This repository consists of scripts for reproducing results in the "Modelling variability in dynamic functional
brain networks using embeddings" manuscript.

## Requirements
- Scripts for preprocessing data depends on the [osl](https://github.com/OHBA-analysis/osl) toolbox. 
- Scripts for training models and analysing results depends on the [osl-dynamics](https://osl-dynamics.readthedocs.io) toolbox, which includes source code for the HIVE model, as well as analysis tools.

## Contents
```data_preprocessing```: This directory contains scripts for preprocessing, coregistration, source reconstruction and fixing sign ambiguity for the three MEG datasets used.

```simulations```: This directory contains scripts for simulation analysis on HIVE.
- ```simulation_1.py```: This script shows how covariance deviations is learnt by the variability encoding block in HIVE.
- ```simulation_2.py```: This script shows how the underlying subpopulation structure is inferred by HIVE.
- ```simulation_3.py```: This script shows HIVE performs more accurate inference than HMM-DE and can make use of increasing amount of heterogeneous data.

```real_data```: This directory contains scripts for training HIVE and HMM-DE, and perform analysis on three real MEG datasets.
- ```wakeman_henson```: This directory contains scripts for training, analysing HIVE and HMM-DE on the Wakeman-Henson dataset.
- ```multi_dataset```: This directory contains scripts for training, analysing HIVE and HMM-DE on combined resting-state data from the MRC MEGUK Nottingham site dataset and the Cam-CAN dataset.
- ```camcan```: This directory contains scripts for training HIVE and HMM-DE on the Cam-CAN dataset resting-state data. There is also a script for performing age prediction from inferred features from both approaches.
