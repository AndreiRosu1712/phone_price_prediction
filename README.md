# Price Range Classification Based on Features

This project uses machine learning techniques to classify a product's price range (`price_range`) based on a set of numerical and categorical features. Several classification models are applied and their performance is compared both on the original dataset and on a reduced version using PCA.

## Input Data

The main input dataset is:
- `trainPrice.csv`: the training dataset containing features and the target label `price_range`.

## Required Libraries

- pandas  
- numpy  
- seaborn  
- matplotlib  
- scikit-learn (model_selection, metrics, decomposition, linear_model, ensemble, neighbors)

## Data Exploration

- Load the dataset and display general information (`shape`, `head`, `info`, `describe`, `isnull().sum()`).
- Separate features into:
  - Categorical features: with fewer than 30 unique values.
  - Numerical features: with 30 or more unique values.
- Visualize feature distributions:
  - Countplots for categorical variables.
  - Histograms and KDE plots for numerical variables, with the mean value highlighted.

## Preprocessing

- Target variable: `price_range`
- Features: all other columns
- Split the dataset into:
  - `X_train`, `X_test`, `y_train`, `y_test` (75% / 25%)

## Dimensionality Reduction

PCA (Principal Component Analysis) is applied to reduce dimensionality while retaining 95% of the explained variance:
- `X_train_reduced`
- `X_test_reduced`

## Models Trained

### Logistic Regression

- Trained on the original dataset
- Trained on the PCA-reduced dataset
- Accuracy and confusion matrices are computed and visualized for both cases

### Random Forest

- Trained on the original dataset
- Trained on the PCA-reduced dataset
- Evaluates multiple `n_estimators` values (10, 50, 100, 200, 300) using `cross_val_score`
- Accuracy and confusion matrices are computed and visualized

### K-Nearest Neighbors (KNN)

- `n_neighbors=5`
- Trained on the original dataset
- Trained on the PCA-reduced dataset
- Accuracy and confusion matrices are computed and visualized

## Evaluation

For each model:
- Accuracy is calculated using `accuracy_score`
- Confusion matrix is displayed using `ConfusionMatrixDisplay`

## Observations and Potential Issues

- `confusion_matrix` is not explicitly imported (should be added: `from sklearn.metrics import confusion_matrix`)
- Variables `knnoriginal` and `knnreduced` used in `ConfusionMatrixDisplay` are not defined correctly; likely intended to be `knn_original` and `knn_reduced`

## Recommendations

- Use a preprocessing pipeline to handle feature scaling
- Normalize features before applying KNN
- Save model results and plots to an output directory
- Fix typographical errors and missing imports for full execution

## Conclusion

This script compares the performance of three classification models (Logistic Regression, Random Forest, KNN) on both the original and PCA-reduced datasets, analyzing the impact of dimensionality reduction on model accuracy.
