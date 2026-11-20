# Heart Disease Mortality Prediction

This project explores machine-learning models for predicting mortality among heart failure patients using clinical records. It compares a local Python workflow with a data-augmentation platform as part of a self-guided learning project.

**This is an educational project, not a medical tool.** 

## Dataset

The dataset contains records for 299 heart failure patients and 11 potential predictors, including age, clinical measurements, and comorbidities. The target, `DEATH_EVENT`, indicates whether a patient died during the study follow-up period.

The original `time` column is excluded from modeling, consistent with the original study. The notebook also notes possible imputation in the source data: 25 `platelets` values equal the column mean, and some `age` values are fractional. The original imputation methods are unknown.

- [UCI Machine Learning Repository dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- [Original study](https://link.springer.com/article/10.1186/s12911-020-1023-5)

## Project workflow

The notebook, `heart-disease-pipeline.ipynb`, covers:

1. Data inspection and basic quality checks.
2. Exploratory analysis of the target, predictors, and relationships between features.
3. A stratified 80/20 train-test split using `random_state=42` to ensure equal distribution of death events in both sets.
4. Model training and hyperparameter search for:
   - Random Forest
   - Ridge Classifier
   - Support Vector Classifier (SVC)
   - XGBoost
5. Evaluation using accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves.
6. Permutation importance for the tuned Random Forest model.

## Model Performance

Cross-validation results (stratified 5-fold):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|----|
| **SVC** | 0.778 ± 0.025 | 0.634 ± 0.035 | 0.753 ± 0.105 | **0.684 ± 0.040** | **0.819 ± 0.050** |
| **RF** | 0.765 ± 0.068 | 0.651 ± 0.130 | 0.648 ± 0.104 | 0.641 ± 0.094 | 0.791 ± 0.051 |
| **Ridge** | 0.753 ± 0.051 | 0.600 ± 0.070 | 0.742 ± 0.106 | 0.659 ± 0.063 | 0.791 ± 0.062 |
| **XGB** | 0.749 ± 0.049 | 0.610 ± 0.093 | 0.688 ± 0.116 | 0.637 ± 0.064 | 0.753 ± 0.041 |

**SVC** achieves the highest F1 and ROC-AUC scores, with stable performance across folds. All models show reasonable recall (0.65–0.75), suggesting they catch a substantial portion of death events, though with varying precision. The models perform similarly, with differences within fold-to-fold variability. These findings are exploratory, especially given the small dataset.

## Requirements and running the notebook

Use Python with Jupyter installed. The notebook uses pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and XGBoost.

Install the packages with:

    python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

Place `heart_failure_clinical_records_dataset.csv` in the project directory, then open `heart-disease-pipeline.ipynb` in Jupyter or Visual Studio Code and run the cells in order. The notebook loads the CSV using a relative path, so its working directory should be the project directory.

## Limitations

This dataset is small and comes from a specific patient population. Results may vary with different data splits, and performance on this dataset does not establish how the models would perform on other patients or in clinical practice. 