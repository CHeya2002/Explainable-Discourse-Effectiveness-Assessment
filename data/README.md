\# Data



This directory documents the datasets used in the experiments. Raw dataset

files are not included in this repository.



\## Dataset



The experiments use the PERSUADE corpus, which contains argumentative essays

written by students and discourse-level annotations.



The project combines data used from PERSUADE 1.0 and PERSUADE 2.0 to construct

the dataset used for discourse effectiveness classification.



Each discourse segment is associated with a discourse type and an

effectiveness label.



\### Discourse Types



The seven discourse types considered in this project are:



\- Position

\- Claim

\- Evidence

\- Counterclaim

\- Rebuttal

\- Concession

\- Concluding Summary



\### Effectiveness Labels



The classification task uses three effectiveness classes:



\- Ineffective

\- Adequate

\- Effective



\## Obtaining the Data



The raw PERSUADE datasets are not redistributed through this repository.



Users should obtain the datasets from their original distribution sources

and comply with the corresponding terms of use and licensing conditions.



Links to the official/original dataset sources will be provided in the main

repository README.



\## Preprocessing and Dataset Integration



After obtaining the required data, preprocessing and dataset integration can

be reproduced using:



`notebooks/01\_data\_preprocessing\_and\_merging.ipynb`



The notebook performs the preprocessing and merging steps required to

construct the dataset used by the subsequent experiments.



\## Data Splitting



The processed data are divided into training, validation, and test sets using

a 70/10/20 split.



Splitting is performed at the essay level rather than independently at the

discourse-segment level. Discourse segments originating from the same essay

are therefore kept within the same partition.



This strategy prevents information leakage between the training, validation,

and test sets.



\## Data Availability



Raw PERSUADE data are intentionally excluded from this GitHub repository.

Only code and documentation required to reconstruct the experimental dataset

from the original sources are provided.

