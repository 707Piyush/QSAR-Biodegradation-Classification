
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

st.set_page_config(
    page_title="QSAR Biodegradation Predictor",
    page_icon="🧪",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "biodeg.csv"
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.pkl"

COLUMNS = [f"descriptor_{i}" for i in range(1, 42)] + ["class"]


@st.cache_data
def load_data():
    data = pd.read_csv(
        DATA_PATH,
        sep=";",
        header=None,
        names=COLUMNS
    )
    return data.drop_duplicates().copy()


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)
    return model, features


df = load_data()
model, feature_columns = load_model()

st.title("🧪 QSAR Biodegradation Predictor")
st.write(
    "A machine learning project that predicts whether a molecule "
    "is Ready Biodegradable (RB) or Not Ready Biodegradable (NRB) "
    "using 41 molecular descriptors."
)

st.caption(
    "Educational prediction tool. Results are model estimates, "
    "not experimental confirmation of biodegradability."
)

overview, eda, performance, prediction = st.tabs([
    "Project Overview",
    "Explore Dataset",
    "Model Performance",
    "Predict Biodegradability"
])

with overview:
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Molecules", len(df))
    col2.metric("Molecular Descriptors", len(feature_columns))
    col3.metric("Exact Duplicate Rows Removed", 3)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Class Distribution")
    counts = df["class"].value_counts().reindex(
        ["RB", "NRB"], fill_value=0
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        legend=False,
        ax=ax
    )
    ax.set_xlabel("Biodegradation Class")
    ax.set_ylabel("Number of Molecules")
    ax.set_title("RB vs NRB")
    st.pyplot(fig)
    plt.close(fig)

    st.write("Class counts:")
    st.dataframe(
        counts.rename("Number of molecules"),
        use_container_width=True
    )

    st.info(
        "RB means Ready Biodegradable. NRB means Not Ready "
        "Biodegradable. The dataset contains 41 descriptors "
        "for each molecule."
    )

with eda:
    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Quality")

    quality1, quality2, quality3 = st.columns(3)
    quality1.metric("Missing Values", int(df.isna().sum().sum()))
    quality2.metric(
        "Remaining Exact Duplicate Rows",
        int(df.duplicated().sum())
    )
    quality3.metric(
        "Descriptor Columns",
        len(feature_columns)
    )

    st.subheader("Descriptor Distribution")

    selected_descriptors = st.multiselect(
        "Choose descriptors to inspect",
        feature_columns,
        default=feature_columns[:4]
    )

    if selected_descriptors:
        fig, ax = plt.subplots(figsize=(10, 5))
        for feature in selected_descriptors:
            sns.histplot(
                df[feature],
                label=feature,
                kde=True,
                element="step",
                stat="density",
                common_norm=False,
                ax=ax
            )
        ax.set_title("Selected Descriptor Distributions")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 8))
    correlation = df[feature_columns].corr()
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        ax=ax
    )
    ax.set_title("Correlation Between Molecular Descriptors")
    st.pyplot(fig)
    plt.close(fig)

with performance:
    st.header("Model Performance Comparison")

    st.write(
        "These results come from the test-set evaluations recorded "
        "during model development. The SVM numbers are from the "
        "initial SVM; its tuned version has not yet been added here."
    )

    comparison = pd.DataFrame({
        "Model": ["Initial SVM", "Tuned Decision Tree", "Tuned Random Forest"],
        "Test Accuracy": [0.8531, 0.8294, 0.8863],
        "RB Precision": [0.7941, 0.7333, 0.8615],
        "RB Recall": [0.7606, 0.7746, 0.7887],
        "RB F1-score": [0.7770, 0.7534, 0.8235],
        "5-Fold CV RB F1": [None, 0.7510, 0.8109]
    })

    st.dataframe(
        comparison.style.format(
            {
                "Test Accuracy": "{:.2%}",
                "RB Precision": "{:.2%}",
                "RB Recall": "{:.2%}",
                "RB F1-score": "{:.2%}",
                "5-Fold CV RB F1": lambda x: "—" if pd.isna(x) else f"{x:.2%}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Test RB F1-score")

    chart_data = comparison.set_index("Model")[["RB F1-score"]]
    st.bar_chart(chart_data)

    st.subheader("Random Forest Feature Importance")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns
    ).sort_values(ascending=False)

    top_features = importance.head(10).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    top_features.plot(kind="barh", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Descriptor")
    ax.set_title("Top 10 Important Molecular Descriptors")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.success(
        "The Random Forest currently has the strongest reported "
        "test-set RB F1-score (82.35%)."
    )

with prediction:
    st.header("Predict Biodegradability")
    st.write(
        "Enter the 41 molecular descriptor values. The initial "
        "values are dataset medians; replace them with the actual "
        "descriptor values for the molecule you want to evaluate."
    )

    medians = df[feature_columns].median()

    with st.form("prediction_form"):
        values = {}

        st.subheader("Molecular Descriptor Inputs")

        for start in range(0, len(feature_columns), 3):
            cols = st.columns(3)

            for col, feature in zip(
                cols,
                feature_columns[start:start + 3]
            ):
                values[feature] = col.number_input(
                    feature,
                    value=float(medians[feature]),
                    format="%.6f",
                    key=f"input_{feature}"
                )

        submitted = st.form_submit_button(
            "Predict Biodegradability",
            type="primary"
        )

    if submitted:
        input_df = pd.DataFrame(
            [[values[feature] for feature in feature_columns]],
            columns=feature_columns
        )

        prediction_result = model.predict(input_df)[0]

        st.subheader("Prediction Result")

        if prediction_result == "RB":
            st.success("Predicted Class: RB — Ready Biodegradable")
        else:
            st.warning("Predicted Class: NRB — Not Ready Biodegradable")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            probability_df = pd.DataFrame({
                "Class": classes,
                "Estimated Probability": probabilities
            })

            st.subheader("Estimated Class Probabilities")
            st.bar_chart(
                probability_df.set_index("Class")
            )

            st.caption(
                "Probabilities are model estimates and may not "
                "be perfectly calibrated."
            )

        st.write(
            "This prediction is only as reliable as the model and "
            "the descriptor values supplied. It does not replace "
            "laboratory biodegradation testing."
        )
