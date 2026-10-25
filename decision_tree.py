
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    make_scorer
)

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

tree = DecisionTreeClassifier(random_state=42)

parameter_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 8, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
rb_f1_scorer = make_scorer(f1_score, pos_label="RB")
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=parameter_grid,
    scoring=rb_f1_scorer,
    cv=cv,
    n_jobs=-1,
    refit=True
)

print("Training and tuning Decision Tree...")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest 5-Fold Cross-Validation RB F1-score:")
print(round(grid_search.best_score_, 4))

print("\nDecision Tree Test Results")
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

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Descriptors:")
print(importance.head(10).to_string())
