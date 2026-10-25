
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

columns = [f"descriptor_{i}" for i in range(1, 42)] + ["class"]

df = pd.read_csv("biodeg.csv", sep=";", header=None, names=columns)

print("Original dataset shape:", df.shape)

df = df.drop_duplicates().copy()

print("Shape after removing exact duplicates:", df.shape)
print("\nClass distribution:")
print(df["class"].value_counts())

X = df.drop(columns=["class"])
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
])

svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

rb_index = list(svm_model.classes_).index("RB")
rb_probabilities = svm_model.predict_proba(X_test)[:, rb_index]
y_test_binary = (y_test == "RB").astype(int)

models = {
    "Baseline": baseline_predictions,
    "SVM": svm_predictions
}

for model_name, predictions in models.items():
    print(f"\n========== {model_name} RESULTS ==========")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("Precision:", round(precision_score(
        y_test, predictions, pos_label="RB", zero_division=0
    ), 4))
    print("Recall:", round(recall_score(
        y_test, predictions, pos_label="RB", zero_division=0
    ), 4))
    print("F1-score:", round(f1_score(
        y_test, predictions, pos_label="RB", zero_division=0
    ), 4))

print("\nSVM ROC-AUC:", round(
    roc_auc_score(y_test_binary, rb_probabilities), 4
))

print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions, zero_division=0))

cm = confusion_matrix(y_test, svm_predictions, labels=["RB", "NRB"])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["RB", "NRB"],
    yticklabels=["RB", "NRB"]
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("SVM Confusion Matrix")
plt.tight_layout()
plt.savefig("svm_confusion_matrix.png", dpi=150)
plt.show()
