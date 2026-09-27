# Explainable Discourse Effectiveness Assessment

This repository contains the implementation associated with a research framework for **explainable discourse effectiveness assessment in student argumentative writing**.

The framework combines Transformer-based discourse classification, Explainable Artificial Intelligence (XAI), and rule-guided narrative generation to support both predictive assessment and interpretable communication of model decisions.

## Overview

The final experimental pipeline consists of four main stages:

1. **Data Preprocessing and Integration**  
   Preparation, cleaning, integration, and partitioning of the discourse-level dataset.

2. **Discourse Effectiveness Classification**  
   Fine-tuning DeBERTa for three-class discourse effectiveness classification.

3. **Explainable AI**  
   Generation of local token-level explanations using LIME and Integrated Gradients.

4. **Narrative-Based Explanation**  
   Transformation of model predictions and XAI evidence into educational narratives using a rule-guided storytelling framework and a locally deployed language model.

## Repository Structure

```text
Explainable-Discourse-Effectiveness-Assessment/
│
├── src/
│   ├── 01_data_preprocessing_and_merging.py
│   ├── 02_deberta_classification.py
│   ├── 03_xai_lime_integrated_gradients.py
│   └── 04_storytelling_generation.py
│
├── data/
│   └── README.md
│
├── figures/
│   └── README.md
│
├── .gitignore
├── requirements.txt
└── README.md
```

The `src/` directory contains the final implementation used to represent the experimental pipeline. Intermediate development notebooks and experimental iterations are not included.

## Dataset

The experiments are based on the **PERSUADE corpus**, which contains argumentative writing produced by students and discourse-level annotations.

The classification task considers three discourse effectiveness labels:

- **Ineffective**
- **Adequate**
- **Effective**

The repository does not redistribute the original dataset. Dataset acquisition and preparation information is provided in `data/README.md`.

Local dataset files are excluded from version control.

## Data Splitting

To reduce information leakage between partitions, the data are divided at the essay level rather than independently at the discourse-segment level.

The final experimental partition follows approximately:

- **70% training**
- **10% validation**
- **20% testing**

The preprocessing and splitting procedure is implemented in:

```text
src/01_data_preprocessing_and_merging.py
```

## Classification Model

The final classifier is based on **DeBERTa-v3-base** and predicts one of the three discourse effectiveness classes.

The training strategy includes mechanisms for handling class imbalance and controlling overfitting, including:

- weighted focal loss;
- learning-rate scheduling;
- weight decay;
- gradient clipping;
- early stopping;
- validation-based model selection.

The classification implementation is provided in:

```text
src/02_deberta_classification.py
```

Trained model checkpoints are not stored in this repository.

## Explainable AI

Two complementary post-hoc explanation methods are applied to the trained classifier:

- **LIME (Local Interpretable Model-agnostic Explanations)** for local surrogate-based token importance;
- **Integrated Gradients (IG)** for gradient-based attribution with respect to the model output.

The resulting explanations provide token-level evidence for interpreting individual model predictions.

The implementation is provided in:

```text
src/03_xai_lime_integrated_gradients.py
```

## Narrative-Based Explanation

The final stage transforms model predictions, class probabilities, XAI evidence, assessment information, and an engine-controlled narrative structure into natural-language educational explanations.

Narratives are generated locally using:

- **Ollama**
- **Qwen2.5 7B**

The language model is used for narrative realization, while the supplied rule-guided structure constrains the information and evidence used during generation.

The implementation is provided in:

```text
src/04_storytelling_generation.py
```

For the final public pipeline, the storytelling stage expects a prepared input file:

```text
data/storytelling_inputs.json
```

This file represents the final engine-controlled input supplied to the narrative-generation stage. Intermediate storytelling development stages are not included in the repository.

## Installation

Python 3.11 is recommended.

Create and activate a virtual environment, then install the required Python dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include PyTorch, Transformers, pandas, scikit-learn, LIME, Captum, Matplotlib, and related utilities.

## Local LLM Setup

Narrative generation requires a local Ollama installation and the Qwen2.5 7B model.

After installing Ollama, obtain the required model with:

```bash
ollama pull qwen2.5:7b
```

The storytelling script communicates with the local Ollama service through:

```text
http://localhost:11434
```

If the local model is unavailable, the implementation contains a deterministic fallback mechanism.

## Running the Pipeline

The final source-code stages are organized in execution order:

```bash
python src/01_data_preprocessing_and_merging.py
python src/02_deberta_classification.py
python src/03_xai_lime_integrated_gradients.py
python src/04_storytelling_generation.py
```

Some stages require their corresponding local data, trained-model, or prepared storytelling inputs. These files are intentionally excluded from version control where appropriate.

Generated outputs and trained models are also excluded from Git tracking.

## Reproducibility

The repository provides the final source implementation of the experimental framework while excluding:

- original dataset files;
- trained model checkpoints;
- generated experimental outputs;
- local LLM model files;
- intermediate development notebooks;
- intermediate storytelling-development stages.

This organization is intended to expose the final methodology and implementation associated with the research work without distributing external datasets or large generated artifacts.

## Research Article

This repository accompanies research on **explainable discourse effectiveness assessment and narrative-based communication of AI explanations**.

The overall framework investigates how Transformer-based discourse classification can be combined with local explainability methods and structured narrative generation to make model decisions more accessible in an educational context.

## Status

The repository contains the final source-code pipeline associated with the research implementation. Documentation and reproducibility information may be updated alongside the corresponding research article.