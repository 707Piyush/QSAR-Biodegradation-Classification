import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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

forest = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

parameter_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", 0.7]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

rb_f1_scorer = make_scorer(f1_score, pos_label="RB")

grid_search = GridSearchCV(
    estimator=forest,
    param_grid=parameter_grid,
    scoring=rb_f1_scorer,
    cv=cv,
    n_jobs=-1,
    refit=True
)

print("Training and tuning Random Forest...")
grid_search.fit(X_train, y_train)

joblib.dump(grid_search.best_estimator_, "random_forest_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")

print("\nModel saved as random_forest_model.pkl")
print("Feature names saved as feature_columns.pkl")

best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest 5-Fold Cross-Validation RB F1-score:")
print(round(grid_search.best_score_, 4))

print("\nRandom Forest Test Results")
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
