
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.metrics import make_scorer

columns = [f"descriptor_{i}" for i in range(1, 42)] + ["class"]

df = pd.read_csv("biodeg.csv", sep=";", header=None, names=columns)
df = df.drop_duplicates().copy()

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])

parameter_grid = [
    {
        "classifier__kernel": ["linear"],
        "classifier__C": [0.1, 1, 10, 100]
    },
    {
        "classifier__kernel": ["rbf"],
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.01, 0.1, 1]
    },
    {
        "classifier__kernel": ["poly"],
        "classifier__C": [0.1, 1, 10],
        "classifier__degree": [2, 3],
        "classifier__gamma": ["scale"]
    }
]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rb_f1_scorer = make_scorer(f1_score, pos_label="RB")

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring=rb_f1_scorer,
    cv=cv,
    n_jobs=-1,
    refit=True
)

print("Training and comparing SVM models...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

scores = best_model.decision_function(X_test)

if best_model.named_steps["classifier"].classes_[1] != "RB":
    scores = -scores

y_test_binary = (y_test == "RB").astype(int)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest 5-Fold Cross-Validation RB F1-score:")
print(round(grid_search.best_score_, 4))

print("\nTest Set Results")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("RB Precision:", round(precision_score(
    y_test, predictions, pos_label="RB", zero_division=0
), 4))
print("RB Recall:", round(recall_score(
    y_test, predictions, pos_label="RB", zero_division=0
), 4))
print("RB F1-score:", round(f1_score(
    y_test, predictions, pos_label="RB", zero_division=0
), 4))
print("ROC-AUC:", round(roc_auc_score(y_test_binary, scores), 4))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nTop 10 Parameter Combinations:")
results = pd.DataFrame(grid_search.cv_results_)
results = results.sort_values("rank_test_score")

print(
    results[
        ["mean_test_score", "std_test_score", "params"]
    ].head(10).to_string(index=False)
)
