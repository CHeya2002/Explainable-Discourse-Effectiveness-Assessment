\# Explainable Discourse Effectiveness Assessment



This repository contains the implementation of an explainable NLP framework

for discourse effectiveness assessment in student argumentative writing.



\## Overview



The framework combines discourse-level classification, explainable artificial

intelligence (XAI), and narrative-based explanation to provide both predictive

assessment and interpretable feedback.



The experimental pipeline consists of four main stages:



1\. Data preprocessing and dataset integration

2\. Discourse effectiveness classification using DeBERTa

3\. Local explanation using LIME and Integrated Gradients

4\. Educational storytelling generation



\## Repository Structure



notebooks/

├── 01\_data\_preprocessing\_and\_merging.ipynb

├── 02\_deberta\_classification.ipynb

├── 03\_xai\_lime\_integrated\_gradients.ipynb

└── 04\_storytelling\_generation.ipynb



\## Dataset



The experiments are based on the PERSUADE corpus for discourse-level

analysis of student argumentative writing.



The classification task considers three discourse effectiveness labels:



\- Ineffective

\- Adequate

\- Effective



Seven discourse types are considered: Position, Claim, Evidence,

Counterclaim, Rebuttal, Concession, and Concluding Summary.



\## Classification Model



DeBERTa is fine-tuned for three-class discourse effectiveness classification.

The training strategy addresses class imbalance and model generalization

through weighted focal loss and regularization techniques.



\## Explainable AI



Two complementary post-hoc explanation methods are used:



\- \*\*LIME\*\* for local surrogate-based feature attribution.

\- \*\*Integrated Gradients (IG)\*\* for gradient-based token attribution.



Their explanations are additionally compared to examine agreement between

local explanation methods.



\## Narrative-Based Explanation



Model predictions and XAI evidence are transformed into structured

natural-language explanations through a rule-guided storytelling pipeline.

The objective is to communicate model behavior in a form that is more

accessible to non-technical educational users.



\## Article



This repository contains the implementation associated with the research

article on explainable discourse effectiveness assessment and

narrative-based communication of AI explanations.



\## Status



The repository is currently under development. Additional documentation,

experimental results, and reproducibility instructions will be added.

