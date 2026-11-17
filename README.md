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
3. An 80/20 train-test split using `random_state=42`.
4. Model training and hyperparameter search for:
   - Random Forest
   - Ridge Classifier
   - Support Vector Classifier (SVC)
   - XGBoost
5. Evaluation using accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves.
6. Permutation importance for the tuned Random Forest model.

The notebook reports that the models have similar cross-validation ROC-AUC scores (about 0.79). XGBoost has the highest reported F1 and recall, but the differences are within fold-to-fold variability. Ejection fraction and serum creatinine appear important to the Random Forest model in the permutation-importance analysis. These findings are exploratory, especially given the small dataset.

## Requirements and running the notebook

Use Python with Jupyter installed. The notebook uses pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and XGBoost.

Install the packages with:

    python -m pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

Place `heart_failure_clinical_records_dataset.csv` in the project directory, then open `heart-disease-pipeline.ipynb` in Jupyter or Visual Studio Code and run the cells in order. The notebook loads the CSV using a relative path, so its working directory should be the project directory.

## Limitations

This dataset is small and comes from a specific patient population. Results may vary with different data splits, and performance on this dataset does not establish how the models would perform on other patients or in clinical practice. The notebook’s cross-validation scores are calculated after hyperparameter tuning on the training set, rather than using nested cross-validation, so they may be optimistic.