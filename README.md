# QSAR Biodegradation Classification 🧪

A Machine Learning project that predicts whether a chemical compound is **Ready Biodegradable (RB)** or **Not Ready Biodegradable (NRB)** using molecular descriptors and supervised learning algorithms.

## 📌 Project Overview

Biodegradation is the process by which microorganisms break down chemical substances. Predicting biodegradability can help in the early assessment of chemicals and their potential environmental impact.

This project uses the QSAR biodegradation dataset to build, evaluate, and compare multiple machine learning classification models.

## 🎯 Objectives

- Analyze and understand the dataset using Exploratory Data Analysis (EDA).
- Train machine learning models to classify chemical compounds.
- Apply cross-validation and hyperparameter tuning.
- Compare model performance using classification metrics.
- Develop an interactive web application using Streamlit.

## 📊 Dataset

- **Total samples:** 1,055
- **Molecular descriptors:** 41
- **Target classes:** RB and NRB
- **Target variable:** Biodegradability class

| Class | Meaning |
|---|---|
| RB | Ready Biodegradable |
| NRB | Not Ready Biodegradable |

The dataset contains numerical molecular descriptors used as input features for prediction.

## 🛠️ Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib
- Git and GitHub

## 🤖 Machine Learning Models

The following models were trained and evaluated:

1. **Support Vector Machine (SVM)** — includes kernel comparison and hyperparameter tuning.
2. **Decision Tree Classifier** — uses a tuned decision tree for classification.
3. **Random Forest Classifier** — combines multiple decision trees to improve predictive performance.

Stratified train-test splitting and five-fold cross-validation were used during model evaluation and tuning.

## 📈 Model Performance

The following results were obtained on the held-out test set.

| Model | Accuracy | RB Precision | RB Recall | RB F1-score |
|---|---:|---:|---:|---:|
| SVM (Linear) | 86.26% | 80.00% | 78.87% | 79.43% |
| Decision Tree | 82.94% | 73.33% | 77.46% | 75.34% |
| Random Forest | **88.63%** | **86.15%** | 78.87% | **82.35%** |

**Best overall test-set performance:** Random Forest, with 88.63% accuracy and an RB F1-score of 82.35%.

These metrics describe performance on this particular test split; results may vary with different data splits or datasets.

## ✨ Features

- Dataset exploration and visualization
- Comparison of machine learning models
- Hyperparameter tuning using cross-validation
- Prediction of biodegradability classes
- Interactive Streamlit interface
- Saved model for reuse without retraining

## 🚀 Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/707Piyush/QSAR-Biodegradation-Classification.git
cd QSAR-Biodegradation-Classification
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 4. Run the web application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`, where you can open the application in your browser.

## 📁 Project Structure

```text
QSAR-Biodegradation-Classification/
├── app.py
├── biodeg.csv
├── model_selection.py
├── decision_tree.py
├── random_forest.py
├── random_forest_model.pkl
├── feature_columns.pkl
├── .gitignore
└── README.md
```

*This structure represents the expected project files; adjust it to match the actual contents of your repository.*

## 🔬 Evaluation Metrics

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** How often predicted RB samples are actually RB.
- **Recall:** How many actual RB samples are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures how well a classifier separates the classes across thresholds.
- **Cross-validation:** Evaluates performance across multiple training and validation folds.

## 🔮 Future Improvements

- Test additional classification algorithms.
- Add more comprehensive visualizations and model explanations.
- Improve the application with molecule-specific descriptor input.
- Evaluate generalization on an independent external dataset.
- Deploy the Streamlit application online.

## 👨‍💻 Author

**Piyush Gupta**

- GitHub: [707Piyush](https://github.com/707Piyush)
- LinkedIn: [Piyush Gupta](https://www.linkedin.com/in/07piyush/)

---

⭐ If you find this project interesting, feel free to explore the repository and share feedback.

**Disclaimer:** This project is intended for educational and research purposes. Predictions should not be treated as definitive evidence of a chemical compound's environmental safety.
