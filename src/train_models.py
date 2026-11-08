import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

from preprocessing import (
    create_preprocessor,
    split_data,
)


MODEL_DIR = "models"
REPORT_DIR = "reports"
PROCESSED_DIR = "data/processed"

RANDOM_STATE = 42
CV_SPLITS = 5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def get_model_searches():

    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "dense": False,
            "params": {
                "classifier__C": [0.01, 0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs"],
            },
        },

        "Decision Tree": {
            "estimator": DecisionTreeClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "dense": False,
            "params": {
                "classifier__max_depth": [
                    None, 5, 10, 15, 20
                ],
                "classifier__min_samples_split": [
                    2, 5, 10
                ],
                "classifier__min_samples_leaf": [
                    1, 2, 5
                ],
                "classifier__criterion": [
                    "gini", "entropy"
                ],
            },
        },

        "KNN": {
            "estimator": KNeighborsClassifier(),
            "dense": False,
            "params": {
                "classifier__n_neighbors": [
                    5, 7, 9, 11, 15
                ],
                "classifier__weights": [
                    "uniform", "distance"
                ],
                "classifier__p": [1, 2],
            },
        },

        "Gaussian Naive Bayes": {
            "estimator": GaussianNB(),
            "dense": True,
            "params": {
                "classifier__var_smoothing": [
                    1e-11,
                    1e-10,
                    1e-9,
                    1e-8,
                    1e-7,
                ],
            },
        },

        "SVM": {
            "estimator": SVC(
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "dense": False,
            "params": {
                "classifier__C": [
                    0.1, 1.0, 10.0
                ],
                "classifier__kernel": [
                    "rbf", "linear"
                ],
                "classifier__gamma": [
                    "scale", "auto"
                ],
            },
        },
    }


def build_pipeline(classifier, dense_output=False):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(
                    dense_output=dense_output
                ),
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def get_positive_probabilities(model, X):

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError(
            "The trained model does not contain positive class 1."
        )

    positive_index = classes.index(1)

    return model.predict_proba(X)[:, positive_index]


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = get_positive_probabilities(
        model,
        X_test,
    )

    return {
        "Accuracy": accuracy_score(
            y_test,
            y_pred,
        ),
        "AUC": roc_auc_score(
            y_test,
            y_prob,
        ),
        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "MCC": matthews_corrcoef(
            y_test,
            y_pred,
        ),
    }


def get_model_filename(model_name):

    filenames = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
        "SVM": "svm.joblib",
    }

    return filenames[model_name]


def save_model_reports(
    model_name,
    model,
    X_test,
    y_test,
):

    safe_name = (
        model_name.lower()
        .replace(" ", "_")
    )

    y_pred = model.predict(X_test)

    y_prob = get_positive_probabilities(
        model,
        X_test,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=[
            "No Purchase",
            "Purchase",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        os.path.join(
            REPORT_DIR,
            f"{safe_name}_classification_report.csv",
        )
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual_No_Purchase", "Actual_Purchase"],
        columns=["Predicted_No_Purchase", "Predicted_Purchase"],
    )

    matrix_df.to_csv(
        os.path.join(
            REPORT_DIR,
            f"{safe_name}_confusion_matrix.csv",
        )
    )

    predictions = pd.DataFrame({
        "Actual_Revenue": y_test.to_numpy(),
        "Predicted_Revenue": y_pred,
        "Purchase_Probability": y_prob,
    })

    predictions.to_csv(
        os.path.join(
            REPORT_DIR,
            f"{safe_name}_test_predictions.csv",
        ),
        index=False,
    )


def train_models():

    X_train, X_test, y_train, y_test = split_data()

    print("\nTraining class counts:")
    print(y_train.value_counts().sort_index())

    print("\nTest class counts:")
    print(y_test.value_counts().sort_index())

    test_data = X_test.reset_index(drop=True).copy()

    test_data["Revenue"] = (
        y_test.reset_index(drop=True).astype(int)
    )

    test_data.to_csv(
        os.path.join(
            PROCESSED_DIR,
            "test_data.csv",
        ),
        index=False,
    )

    print("\nHeld-out test dataset saved.")

    cv = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model_searches = get_model_searches()

    final_results = []
    cv_results = []
    best_parameters = {}

    for model_name, configuration in model_searches.items():

        print("\n" + "=" * 75)
        print(f"Searching: {model_name}")
        print("=" * 75)

        pipeline = build_pipeline(
            classifier=configuration["estimator"],
            dense_output=configuration["dense"],
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=configuration["params"],
            scoring={
                "roc_auc": "roc_auc",
                "f1": "f1",
                "precision": "precision",
                "recall": "recall",
                "accuracy": "accuracy",
            },
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            error_score="raise",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            search.fit(X_train, y_train)

        best_model = search.best_estimator_

        print("\nBest cross-validation ROC AUC:")
        print(round(search.best_score_, 5))

        print("\nBest parameters:")
        print(search.best_params_)

        metrics = evaluate_model(
            best_model,
            X_test,
            y_test,
        )

        metrics["Model"] = model_name
        metrics["CV ROC AUC"] = search.best_score_

        final_results.append(metrics)

        best_parameters[model_name] = {
            key: (
                value.item()
                if isinstance(value, np.generic)
                else value
            )
            for key, value in search.best_params_.items()
        }

        cv_results.append({
            "Model": model_name,
            "CV ROC AUC Mean": search.best_score_,
            "CV ROC AUC Std": search.cv_results_[
                "std_test_roc_auc"
            ][search.best_index_],
            "CV F1 Mean": search.cv_results_[
                "mean_test_f1"
            ][search.best_index_],
            "CV Precision Mean": search.cv_results_[
                "mean_test_precision"
            ][search.best_index_],
            "CV Recall Mean": search.cv_results_[
                "mean_test_recall"
            ][search.best_index_],
            "CV Accuracy Mean": search.cv_results_[
                "mean_test_accuracy"
            ][search.best_index_],
        })

        model_path = os.path.join(
            MODEL_DIR,
            get_model_filename(model_name),
        )

        joblib.dump(
            best_model,
            model_path,
        )

        save_model_reports(
            model_name,
            best_model,
            X_test,
            y_test,
        )

        print(f"\nSaved model: {model_path}")
        print("Held-out test metrics:")
        print(metrics)

    results_df = pd.DataFrame(final_results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC",
            "CV ROC AUC",
        ]
    ].sort_values(
        by="AUC",
        ascending=False,
    )

    results_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "model_comparison.csv",
        ),
        index=False,
    )

    cv_df = pd.DataFrame(cv_results)

    cv_df.to_csv(
        os.path.join(
            REPORT_DIR,
            "cross_validation_results.csv",
        ),
        index=False,
    )

    with open(
        os.path.join(
            REPORT_DIR,
            "best_hyperparameters.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_parameters,
            file,
            indent=4,
        )

    experiment_info = {
        "random_state": RANDOM_STATE,
        "test_size": 0.20,
        "cross_validation": "StratifiedKFold",
        "cross_validation_splits": CV_SPLITS,
        "selection_metric": "ROC AUC",
        "positive_class": 1,
        "training_records": len(X_train),
        "test_records": len(X_test),
    }

    with open(
        os.path.join(
            REPORT_DIR,
            "experiment_configuration.json",
        ),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            experiment_info,
            file,
            indent=4,
        )

    print("\n" + "=" * 85)
    print("FINAL HELD-OUT TEST COMPARISON")
    print("=" * 85)
    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nAll models, metrics and reports saved successfully.")


if __name__ == "__main__":
    train_models()