# ML implementation for "Machine Learning-Guided Dose-Time Optimization and Experimental Validation Enhance Coumarin Therapeutics in Oncology"

>  Please note that this repository is mainly meant for the reviewers and may be closed once publication is accepted.(or at the very least the dataset will be removed)

This GitHub repository is meant for all the machine learning implementation relevant to 
"Machine Learning-Guided Dose-Time Optimization and Experimental Validation Enhance Coumarin Therapeutics in Oncology"
as well as the python scripts used for generating some of the figures.

In order to run all scripts in this repository you will need the following python modules installed 
on your environment: 
- sklearn (1.7.2)
- xgboost (3.1.1)
- pandas (2.3.3)
- numpy (2.3.4)
- os
- sys
- joblib (1.5.2)
- matplotlib (3.10.7)
- graphviz (0.21)

> *The python version is 3.13.7

All the code regarding the implementations can be found in the `main.ipynb` notebook, python scripts
for each of the notebook cells can be found at the `src` directory.

## Data
The `Total_Data.csv` file presents the raw data, and the `Processed_data.csv` file is the 
post-processing data.

## Pre-processing
The `Preprocessing.py` script implements all the pre-processing steps and creates the 
`Processed_data.csv` file. These steps include:
- GMM reliability filtering.
- Encoding of the CancerType and Coumarin features.
- Applying the `Time` constraints for Auraptene and the other coumarins

### The GMM
A new data frame with 2 columns, `Cancer Type` and `Sample Count` is created and the GMM is fitted on
the `Sample Count` columns of this data frame with `n_components = 2` and `random_state = 42`.
Afterwards the inclusion threshold is calcuated as the average of the 2 components` means, Then a
new column is added to the data frame that presents the reliability of the Cancer type. Finally,
the cancers that were deemed unreliable are excluded from the final dataset. It must be noted that
this component of the pre-processing pipeline exists only because our dataset is rather small, with
the data distributed over 18 different cancer types .

## Training results
Within the `TrainingResults` directory you can find sub-directories for each model. In each one of
these sub-directories you can find the best hyper-parameters found through grid searching, scores
from the cross-validation of the model, and a `joblib` file storing the model object.

## Model training and grid searching
Each model has its own notebook cell (or python script in `src`). when running each of these scripts
you'll be prompted for grid searching, if you respond with `yes`, grid searching will 
be done for the model and the best hyper-parameters will be stored in `TrainingResults` directory
and then the program stops; if you respond with `no`, the program will load the best 
hyper-parameters and fit the model on the processed data. Then the cross-validation of the model 
will done and the predictions by the model will be printed in your terminal.

## Reproducibility
In order to reproduce all our results, you can start by cloning the repository into a folder on your
system and opening the folder in your IDE or editor of choice(I recommend Visual Studio Code). 
Then delete the `Training Results` directory and the `Processed_data.csv` file. Start running the cells in the 
notebook in order, or run the python scripts in the `src` directory in the order that they appear 
inside the notebook. When running the scripts for each model, you will be prompted on whether or not you 
intend to perform grid searching, for the first iteration respond with `yes` so that the best 
parameters are found and stored, then run the script again and respond with `no` so that model 
evaluation and training begins. Once you've finished running all the scripts, you should see that 
the new `TrainingResults` directory and all its sub-directories and files perfectly match the ones 
on the repository. The scripts generating figures 2,3-A,3-B and 5 are also available and can be run
so that the figures are regenerated.