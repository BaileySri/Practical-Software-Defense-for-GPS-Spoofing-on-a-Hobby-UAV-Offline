# Practical Software Defense for GPS Spoofing on a Hobby UAV


The Jupyter Notebooks made available in this repository show the offline processing that was performed to determine parameters for testing and parse results.

## Notebooks
* [False Positive Testing](/notebooks/False%20Positive%20Rates.ipynb)
    * False positives for the disagreement in OF, Gyroscope, and GPS over missions Benign 1-5
* [Loiter Missions](/notebooks/Spoofing%20Loiter.ipynb)
    * Looks at the performance of the defense for the Onestep and Multistep Loiter missions
* [Flights](/notebooks/Spoofing%20Flights.ipynb)
    * Looks at the performance of the defense for missions F-Subtle 1-4 using OF and Gyroscope
* [Choi Control Invariant](/notebooks/Choi%20Model.ipynb)
    * Recreate the Choi control invariant to run over data logs using parameters derived from their [MATLAB scripts](/src/choi/Matlab_scripts/)

## Setup Environment
Assuming current directory is the base of the project repository (change NAME to desired conda environment name)
1. conda create env --name NAME --file environment.yml
2. pip install --editable .

## Project Organization
------------

    ├── README.md   <- README
    ├── data
    │   └── interim <- Intermediate data that has been transformed
    │
    ├── notebooks   <- Jupyter notebooks.
    │
    ├── setup.py    <- Make this project pip installable with `pip install -e`
    ├── src         <- Source code for use in this project.
    │   └── confirmation        <- Helper functions to transform and visualize data
    │   │  ├── visualize.py
    │   │  └── process.py
    │   └── choi                <- Choi Control Invariant files
    │      ├── Matlab_Scripts
    │      └── README.md
    │
    ├── environment.yml     <- The requirements to recreate the conda environment this notebook runs under
    │
    └── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
                                generated with `pip freeze --format=freeze > requirements.txt`

--------
