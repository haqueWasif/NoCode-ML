# NoCodeML Studio 🧠✨

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Pipeline](https://img.shields.io/badge/Architecture-sklearn.Pipeline-green)
![License](https://img.shields.io/badge/License-MIT-green)

**NoCodeML Studio** is an end-to-end machine learning platform designed for rapid prototyping and educational experimentation. 

Unlike standard dashboards, this application is built upon robust **Scikit-Learn Pipelines**. It ensures that every preprocessing step—from imputation to encoding—is stateful and leakage-proof. Users can visually build a model, tune it, and export a production-ready Python script or serialized model file without writing code.

---

## 🚀 Key Features

### 📊 Data & Profiling
* **Broad Format Support:** Upload CSV, Excel, JSON, Parquet, or TSV files.
* **Deep Profiling:** Integration with `ydata-profiling` for comprehensive exploratory data analysis (EDA).
* **Visualization:** Interactive correlation heatmaps and distribution plots.

### 🛠️ Advanced Preprocessing
* **Leakage-Free Architecture:** All transformations are fitted on training data and applied consistently to test/inference data.
* **Custom Scripting:** A secure sandbox to write custom Python logic for feature engineering (e.g., `df['ratio'] = df['col_a'] / df['col_b']`).
* **Handling Missing Data:** Strategies include Mean, Median, KNN, Random Sampling, and Constant value imputation.
* **Data Cleaning:** Outlier clipping (IQR method) and automatic binning.
* **Scaling & Encoding:** Supports Standard, MinMax, Robust, and MaxAbs scalers; OneHot and Ordinal encoders.
* **Imbalance Handling:** Integrated **SMOTE** (Synthetic Minority Over-sampling Technique) for classification tasks.

### 🤖 Modeling & Training
* **Multi-Task:** Supports both **Classification** and **Regression**.
* **Algorithm Suite:** Includes Linear/Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, and Naive Bayes.
* **Ensembles:** Build **Voting** and **Stacking** ensembles via the UI.
* **Time-Series Mode:** Time-aware train/test splitting for temporal datasets.

### ⚡ Tuning & MLOps
* **Hyperparameter Optimization:** Automated Grid Search and Random Search.
* **Code Generation:** Export the entire visual pipeline as a clean, executable Python script.
* **Model Export:** Download the trained pipeline as a `.pkl` file (using `dill` for robust serialization).

---

## 📂 Application Workflow

The application guides users through a logical 6-step workflow:

1.  **📊 Visualization:** Preview raw data, view statistics, and generate profile reports.
2.  **⚙️ Preprocessing:** Configure the pipeline (imputation, encoding, scaling, SMOTE, PCA).
3.  **🧠 Training:** Select algorithms, configure ensembles, and execute training.
4.  **📈 Evaluation:** Analyze performance using Confusion Matrices, ROC Curves, and Residual plots.
5.  **⚡ Tuning:** Optimize hyperparameters using Cross-Validation.
6.  **🔮 Inference:** Upload new datasets to generate predictions using the frozen pipeline.

---

## 💻 Installation

### Prerequisites
* Python 3.9 or higher

### 1. Clone the Repository
```bash
git clone [https://github.com/haqueWasif/NoCode-ML.git](https://github.com/haqueWasif/NoCode-ML.git)
cd NoCode-ML

```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt

```

### 3. Run the Application

```bash
streamlit run test.py

```

---

## 📦 Requirements

Create a `requirements.txt` file with the following dependencies to ensure all features (including CatBoost and LightGBM) work correctly:

```text
streamlit
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
catboost
imbalanced-learn
ydata-profiling
dill

```

---

## 👥 Meet the Team

We would like to acknowledge the hard work and dedication of the contributors who made this project possible.

| Contributor | GitHub Profile |
| --- | --- |
| <img src="https://github.com/haqueWasif.png?size=50" width="50" style="border-radius: 50%;"/> **Wasif Haque** | [@haqueWasif](https://github.com/haqueWasif/) |
| <img src="https://github.com/mdjahirulislam56.png?size=50" width="50" style="border-radius: 50%;"/> **Md Jahirul Islam** | [@mdjahirulislam56](https://github.com/mdjahirulislam56) |
| <img src="https://github.com/Saifuddin-Yasir.png?size=50" width="50" style="border-radius: 50%;"/> **Saifuddin Yasir** | [@Saifuddin-Yasir](https://github.com/Saifuddin-Yasir) |

---

## ⚠️ Disclaimer

This tool allows the execution of custom Python code via the "Custom Feature Engineering" tab. While powerful, this feature uses Python's `exec()` function. **Do not deploy this application on a public server without adding proper sandboxing or disabling the custom script feature.**

---

## © Copyright

**Copyright © 2026 Wasif Haque, Md Jahirul Islam, Saifuddin Yasir.**

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software in accordance with the license terms.

```
