# Heart Disease Mortality Prediction

This project explores machine-learning models for predicting mortality among heart failure patients using clinical records. It is a self-guided learning project, with a planned follow-up that compares this baseline against models trained with synthetic data added by a data-augmentation platform.

**This is an educational project, not a medical tool.**

## Dataset

The dataset contains records for 299 heart failure patients: 11 predictors (age, clinical measurements, and comorbidities), a follow-up time column, and the target. The target, `DEATH_EVENT`, indicates whether a patient died during the follow-up period (96 of 299 patients, about 32%).

The follow-up `time` column is excluded from modeling. The original machine-learning study of this dataset (Chicco & Jurman, 2020) also excluded it from its main experiments, and then showed in a separate analysis that it is highly informative. Here it is excluded because it would not be known at the time a prediction is made.

The notebook also notes possible imputation in the source data: 25 `platelets` values equal the column mean, and some `age` values are fractional. The original imputation methods are unknown, so other imputed values may exist undetected.

- [UCI Machine Learning Repository dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- [Chicco & Jurman (2020), machine-learning study of this dataset](https://link.springer.com/article/10.1186/s12911-020-1023-5)

## Project workflow

The notebook, `heart-disease-pipeline.ipynb`, covers:

1. Data inspection and basic quality checks.
2. Exploratory analysis of the target, predictors, and relationships between features.
3. A stratified 80/20 train-test split (`random_state=42`) so that death events make up the same share of both sets.
4. Model training and hyperparameter search (grid search with stratified 5-fold cross-validation, optimizing F1, on the training data only) for:
   - Random Forest
   - Ridge Classifier
   - Support Vector Classifier (SVC)
   - XGBoost

   Class weighting (`class_weight="balanced"`, or `scale_pos_weight` for XGBoost) was included in the search to counter the class imbalance, and it substantially improved recall.
5. Model comparison using 5-fold cross-validation on the training data (mean and standard deviation of accuracy, precision, recall, F1, and ROC-AUC), followed by a single evaluation on the held-out test set with confusion matrices and ROC curves.
6. Permutation importance for the tuned Random Forest model.

## Model Performance

**Cross-validation results (stratified 5-fold, mean ± std, training data):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|----|
| SVC | 0.778 ± 0.025 | 0.634 ± 0.035 | 0.753 ± 0.105 | 0.684 ± 0.040 | 0.819 ± 0.050 |
| RF | 0.765 ± 0.068 | 0.651 ± 0.130 | 0.648 ± 0.104 | 0.641 ± 0.094 | 0.791 ± 0.051 |
| Ridge | 0.753 ± 0.051 | 0.600 ± 0.070 | 0.742 ± 0.106 | 0.659 ± 0.063 | 0.791 ± 0.062 |
| XGB | 0.749 ± 0.049 | 0.610 ± 0.093 | 0.688 ± 0.116 | 0.637 ± 0.064 | 0.753 ± 0.041 |

**Held-out test set results (60 records, 19 death events):**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|----|
| XGB | 0.733 | 0.565 | 0.684 | 0.619 | 0.793 |
| RF | 0.700 | 0.520 | 0.684 | 0.591 | 0.792 |
| Ridge | 0.700 | 0.526 | 0.526 | 0.526 | 0.757 |
| SVC | 0.667 | 0.476 | 0.526 | 0.500 | 0.693 |

### Interpretation

None of the models predicts death events reliably. ROC-AUC is about 0.69-0.82 across both evaluations, and precision stays modest (roughly 0.48-0.65) even with class weighting, which raised recall. For context, the original study's Random Forest on all clinical features (without follow-up time) reached about 0.74 accuracy and 0.80 ROC-AUC averaged over 100 random splits, which is in the same range as the results here.

The two evaluations rank the models differently: SVC is highest in cross-validation but lowest on the test set. The differences between models are smaller than the uncertainty in either evaluation, so the models are best treated as tied. The test set is unbiased but small (one death event is worth about 5 points of recall), while the cross-validation scores are slightly optimistic because hyperparameters were tuned on the same folds. SVC's training and cross-validated scores are nearly identical, so its lower test score looks like sampling noise combined with picking the top model among near-ties, not overfitting.

Permutation importance on the tuned Random Forest points to a few numeric features (serum creatinine, age, ejection fraction, serum sodium) carrying most of the signal, while the binary features contributed almost nothing. This agrees with the original study, whose Random Forest ranked serum creatinine, ejection fraction, and age as the top three. In this notebook, the order within the top group changed between train/test splits and should not be over-interpreted.

These findings are exploratory. The dataset is small, most features carry little signal, and the strongest predictor (follow-up time) was excluded.

## Planned next step

Compare these baseline results against models trained with synthetic data added, evaluating only on real held-out records, and generating synthetic data from the training records only so no test information leaks in.

## Limitations

This dataset is small (299 patients) and comes from a specific population: heart failure patients at two hospitals in Faisalabad, Pakistan, in 2015, all with left ventricular systolic dysfunction. With only 60 test records, results change noticeably with different data splits, and performance on this dataset does not establish how the models would perform on other patients or in clinical practice.