# ShopSense — Online Shoppers Purchasing Intention

A Streamlit evaluation dashboard for five trained binary-classification models using the **UCI Online Shoppers Purchasing Intention** dataset.

## 1. Problem statement

The objective is to predict whether an online browsing session results in a purchase (`Revenue = 1`) or no purchase (`Revenue = 0`) using session-level behavioural and contextual attributes. The project compares five classification algorithms and evaluates them using six required metrics.

## 2. Dataset description

- **Dataset:** Online Shoppers Purchasing Intention
- **Source:** UCI Machine Learning Repository, dataset 468
- **Source URL:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention
- **Records:** 12,330
- **Predictors:** 17 (10 numerical and 7 categorical)
- **Target:** `Revenue` (binary)
- **Class distribution:** approximately 84.5% no-purchase and 15.5% purchase sessions.

### Features

Numerical: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `PageValues`, `SpecialDay`.

Categorical: `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

## 3. GitHub repository link

https://github.com/Preetham-github/online_shoppers_ml

## 4. Models used and evaluation metrics

The five required classifiers are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Final held-out test comparison

These values are from the completed training run and should be treated as the experiment's reported held-out test benchmark.

| ML model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8451 | 0.8949 | 0.5000 | 0.7382 | 0.5962 | 0.5194 |
| Decision Tree | 0.8337 | 0.9231 | 0.4795 | 0.8560 | 0.6147 | 0.5546 |
| KNN | 0.8723 | 0.8329 | 0.7769 | 0.2461 | 0.3738 | 0.3905 |
| Gaussian Naive Bayes | 0.3094 | 0.7626 | 0.1773 | 0.9503 | 0.2989 | 0.1370 |
| Support Vector Machine | 0.8682 | 0.9017 | 0.5562 | 0.7382 | 0.6344 | 0.5643 |

### Five-fold cross-validation summary

| Model | Mean ROC AUC | ROC AUC Std | Mean F1 | Mean Precision | Mean Recall | Mean Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9056 | 0.0065 | 0.6212 | 0.5245 | 0.7621 | 0.8559 |
| Decision Tree | 0.9204 | 0.0084 | 0.6334 | 0.5047 | 0.8532 | 0.8469 |
| KNN | 0.8480 | 0.0067 | 0.3739 | 0.7909 | 0.2451 | 0.8732 |
| Gaussian Naive Bayes | 0.7835 | 0.0169 | 0.3099 | 0.1843 | 0.9725 | 0.3294 |
| Support Vector Machine | 0.9125 | 0.0041 | 0.6553 | 0.5726 | 0.7667 | 0.8750 |

### Observations

| Model | Observation |
|---|---|
| Logistic Regression | Provides a competitive linear baseline, with test AUC 0.8949 and F1 0.5962. |
| Decision Tree | Achieves the highest test ROC AUC (0.9231) and recall (0.8560), making it effective at identifying purchasing sessions, although its precision is more modest. |
| KNN | Achieves the highest accuracy (0.8723) and precision (0.7769), but its recall (0.2461) is low; it misses many actual purchases. |
| Gaussian Naive Bayes | Achieves high recall (0.9503), but low precision (0.1773) and accuracy (0.3094), indicating a high false-positive rate. |
| Support Vector Machine | Achieves the highest test F1 (0.6344) and MCC (0.5643), offering the strongest overall balance among these reported test metrics. |

**Overall winner:** SVM is the preferred overall model when balancing precision and recall using F1 and MCC. Decision Tree is a strong alternative when ROC AUC and recall are the priority. Model selection ultimately depends on the cost of false positives versus false negatives.

## 5. Training methodology

- A stratified 80:20 train/test split was used with `random_state=42`.
- Hyperparameter search used five-fold `StratifiedKFold` cross-validation on the training partition.
- Each model was built as a scikit-learn pipeline containing preprocessing and a classifier.
- Numerical features: median imputation followed by standard scaling.
- Categorical features: most-frequent imputation followed by one-hot encoding with unknown categories ignored.
- Final metrics were calculated on the held-out test partition.
- The application loads fitted model pipelines. It does not train or tune models on uploaded evaluation data.

### Best parameters recorded during the reported search

- Logistic Regression: `C=0.1`, `solver='lbfgs'`.
- Decision Tree: `criterion='entropy'`, `max_depth=5`, `min_samples_leaf=5`, `min_samples_split=2`.
- KNN: `n_neighbors=15`, `p=1`, `weights='distance'`.
- Gaussian Naive Bayes: `var_smoothing=1e-7`.
- SVM: use the exact best-parameter entry from the generated `best_hyperparameters.json`; do not infer or manually invent it.

## 6. Streamlit application

The application provides:

- CSV upload for a labelled evaluation dataset.
- Saved-model availability checks and model loading.
- Schema and target validation.
- Live evaluation of all five models on the same uploaded rows.
- Accuracy, ROC AUC, precision, recall, F1 and MCC.
- Model comparison chart and benchmark results.
- Confusion matrix, ROC curve and classification report.
- Session-level predictions and CSV export.
- Project guide describing the dataset, features, methodology and reported training results.

The uploaded CSV must contain all 17 feature columns and the `Revenue` target. The target accepts `0/1` or `False/True`. Column order may differ because the app reorders columns to match the fitted preprocessing pipeline. Missing or unexpected feature columns are rejected. Both target classes must be present to calculate ROC AUC.

## 7. Repository structure

Keep the project organized as follows (adjust paths to match the actual repository):

```text
online_shoppers_ml/
├── README.md
├── requirements.txt
├── test_data.csv
├── src/
│   ├── app.py
│   ├── train_models.py
│   ├── preprocessing.py
│   └── models/
│       ├── logistic_regression.joblib
│       ├── decision_tree.joblib
│       ├── knn.joblib
│       ├── gaussian_naive_bayes.joblib
│       └── svm.joblib
├── data/
│   ├── raw/
│   │   └── online_shoppers.csv
│   └── processed/
│       └── test_data.csv
└── reports/
    ├── model_comparison.csv
    ├── cross_validation_results.csv
    └── best_hyperparameters.json
```

The Streamlit app searches for model artifacts in the project-root `models/` folder first and then in `src/models/`. Keep all five `.joblib` files together in one of those locations.

## 8. Installation and execution

Use the same scikit-learn version that was used to serialize the model artifacts. Joblib files containing scikit-learn estimators are not guaranteed to be compatible across versions.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

To retrain models (this is separate from running the Streamlit app):

```bash
python src/train_models.py
```

## 9. Deployment

1. Push the source code, README, requirements, test CSV and all five model artifacts to GitHub.
2. Confirm that the repository is accessible to the evaluator.
3. Open Streamlit Community Cloud and create a new app from the repository.
4. Set the entry point to `src/app.py` (or `app.py` if you move it to the repository root).
5. Confirm that model paths resolve in the deployed repository.
6. Test the deployed app with the labelled test CSV.
7. Add the final repository and live application URLs to the assignment PDF.

## 10. Limitations and responsible interpretation

- The target is imbalanced, so accuracy alone is not sufficient for model selection.
- The metrics describe the particular held-out test split and should not be interpreted as guaranteed real-world performance.
- Check for exact duplicate or near-duplicate records across train and test partitions before making strong generalization claims.
- The default classifier decision threshold is used for precision, recall, F1 and MCC. Any threshold tuning must use training/validation data, not the held-out test set.
- Predictions are statistical estimates, not guarantees that a visitor will purchase.
