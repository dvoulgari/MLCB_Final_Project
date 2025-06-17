# Extending Immune Cell Type Classification with Modern ML

This repository contains the code and resources for a project that extends the work of Aybey et al. (2023) on immune cell type signature discovery and Random Forest classification. We aim to benchmark alternative machine learning models (XGBoost, LightGBM) and apply interpretability techniques (SHAP) to enhance the original approach.
---
# Project Overview
The original paper utilized specific gene signatures and a Random Forest classifier for accurate immune cell type annotation from single-cell RNA sequencing (scRNA-seq) data. Our project investigates whether other tree-based models can offer superior performance or efficiency and seeks to provide deeper insights into the classification process using model interpretability tools.
---
# Data
All datasets used in this project are publicly available and are sourced from the original paper's dedicated GitHub repository:
https://github.com/ba306/immune-cell-signature-discovery-classification-paper
---
# Technical Approach
We're primarily using R & Python for this project.
---
# Key Libraries:

pandas & numpy: For data handling and manipulation.
xgboost & lightgbm: For implementing our alternative classification models.
scikit-learn: For various machine learning utilities, including performance metrics.
shap: For model interpretability to understand gene contributions to classifications.
matplotlib & seaborn: For data visualization.
---
