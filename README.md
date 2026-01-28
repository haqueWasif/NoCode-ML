# NoCode-ML

NoCode-ML is an end-to-end machine learning platform built with Streamlit that allows users to create, train, tune, evaluate, and deploy machine learning models without writing code. It follows industry best practices such as leakage-safe preprocessing, proper train-test splitting, cross-validation, hyperparameter tuning, and explainable pipelines.

The project is designed for students, analysts, and practitioners who want to experiment with machine learning workflows interactively while still relying on robust Scikit-learn pipelines under the hood.

---

## Features

* Upload CSV or Excel datasets
* Automatic data type detection
* Interactive data visualization and profiling
* Leakage-safe preprocessing pipelines
* Advanced feature engineering:

  * Log transformation
  * Binning
  * Date feature extraction
  * Polynomial features
  * PCA
* Multiple imputation strategies
* Categorical encoding (One-Hot, Ordinal)
* Imbalanced data handling with SMOTE
* Support for both Classification and Regression
* Time series–aware training mode
* Train multiple models in parallel
* Ensemble models (Voting and Stacking)
* Automated hyperparameter tuning (Grid Search and Random Search)
* Model evaluation with metrics and plots
* End-to-end inference pipeline for new data

---
## Supported Models

### Classification

* Logistic Regression
* Random Forest Classifier
* XGBoost Classifier
* Support Vector Machine
* K-Nearest Neighbors
* Naive Bayes
* Voting Classifier
* Stacking Classifier

### Regression

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor
* Support Vector Regressor
* K-Nearest Neighbors Regressor
* Ridge, Lasso, ElasticNet
* Voting Regressor
* Stacking Regressor

---

## Project Structure

The application is organized into six tabs:

1. Visualization
   Data preview, summary statistics, correlation heatmaps, distributions, and deep profiling using ydata-profiling.

2. Preprocessing
   Column selection, feature engineering, encoding, scaling, imputation, PCA, polynomial features, SMOTE, and train-test splitting.

3. Training
   Train multiple models and ensembles using a consistent preprocessing pipeline.

4. Evaluation
   Compare model performance with metrics, plots, confusion matrices, and reports.

5. Tuning
   Perform automated hyperparameter optimization using GridSearchCV or RandomizedSearchCV.

6. Inference
   Make predictions on new data using the trained pipeline via CSV upload or manual input.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/NoCode-ML.git
cd NoCode-ML
```

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run test.py
```

---

## Requirements

* Python 3.9+
* streamlit
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* xgboost
* ydata-profiling
* imbalanced-learn (optional, for SMOTE)

---

## Design Philosophy

NoCode-ML prioritizes correctness and reproducibility over shortcuts. All preprocessing is fitted only on training data and applied consistently during evaluation and inference. Feature engineering steps are modular and explicitly controlled by the user, making the system transparent and extensible.

---

## Disclaimer

This project is intended for learning, experimentation, and rapid prototyping. For production deployment, additional validation, monitoring, and security considerations are required.

---

## Author

Wasiful Haque, Md. Jahirul Islam, Saifuddin Yasir

---
