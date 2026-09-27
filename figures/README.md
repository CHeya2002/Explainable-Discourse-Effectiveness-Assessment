\# Figures and Reproducible Outputs



This directory is intended for figures and visual outputs generated from the

experimental pipeline.



The figures used in the associated research article are derived from the

classification, explainability, and narrative-generation stages described

in the project notebooks.



\## Data and Preprocessing Outputs



Generated from:



`notebooks/01\_data\_preprocessing\_and\_merging.ipynb`



Expected outputs include descriptive statistics and distributions associated

with the processed discourse-effectiveness dataset.



\## DeBERTa Classification Results



Generated from:



`notebooks/02\_deberta\_classification.ipynb`



Expected outputs include:



\- Training and validation performance

\- Classification metrics

\- Confusion matrix

\- Class-level performance for Ineffective, Adequate, and Effective discourse



\## Explainable AI Outputs



Generated from:



`notebooks/03\_xai\_lime\_integrated\_gradients.ipynb`



Expected outputs include:



\- LIME token-level feature contributions

\- Integrated Gradients token attributions

\- Comparison of LIME and Integrated Gradients explanations

\- Explanation agreement information for selected discourse examples



The XAI analysis considers representative examples across discourse types and

effectiveness classes.



\## Narrative-Based Explanation Outputs



Generated from:



`notebooks/04\_storytelling\_generation.ipynb`



Expected outputs include structured narrative explanations generated from

classification and XAI evidence.



The narrative stage uses model predictions, confidence information, and

explanation evidence to construct reader-oriented explanations for selected

cases.



\## Reproducibility



Figures may exhibit minor visual differences depending on library versions,

hardware, and rendering environment. Numerical results may also vary slightly

when stochastic model-training procedures are repeated.



The main repository README provides the recommended software environment and

execution order.

