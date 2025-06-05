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

## Models Trained and Results

### Logistic Regression

- Accuracy on original data: **61.8%**
- Accuracy on PCA-reduced data: **80.2%**

**Observation:** Logistic Regression greatly benefits from dimensionality reduction, with a significant accuracy gain of over 18%. This suggests multicollinearity or irrelevant features might be affecting performance on raw data.

### Random Forest

- Accuracy on original data: **85.2%**
- Accuracy on PCA-reduced data: **91.0%**

**Cross-validation accuracy (mean):**
- Original data: up to **87.4%** (best with 200 estimators)
- PCA-reduced data: up to **92.3%** (best with 300 estimators)

**Observation:** Random Forest performs strongly even on raw features. However, PCA still provides a marginal gain (~6%), showing that dimensionality reduction can enhance ensemble methods further when many features are present.

### K-Nearest Neighbors (KNN, k=5)

- Accuracy on original data: **91.0% – 92.2%**
- Accuracy on PCA-reduced data: **90.8% – 91.6%**

**Observation:** KNN achieves the best accuracy on the original data without PCA. Applying PCA results in slightly lower performance, likely due to loss of neighborhood structure in transformed space.

## Evaluation

For each model:
- Accuracy is calculated using `accuracy_score`
- Confusion matrix is displayed using `ConfusionMatrixDisplay`

## Observations and Potential Issues

- `confusion_matrix` is not explicitly imported (`from sklearn.metrics import confusion_matrix` is missing).
- `knnoriginal` and `knnreduced` are undefined; they should be `knn_original` and `knn_reduced` respectively when plotting confusion matrices.
- PCA reduced the dimensionality effectively, but its impact varies per model: it improved Logistic Regression substantially, slightly helped Random Forest, and slightly hindered KNN.

## Recommendations

- Use a preprocessing pipeline to handle feature scaling (especially important for KNN).
- Normalize or standardize features before applying distance-based models.
- Include `GridSearchCV` for hyperparameter optimization.
- Automatically log evaluation metrics and visualizations.
- Consider additional models (SVM, Gradient Boosting) for comparison.

## Conclusion

Random Forest and KNN classifiers provide strong accuracy on this classification task. Logistic Regression, though less performant on raw features, shows substantial improvement with PCA, indicating that dimensionality reduction is beneficial for linear models. KNN performs best without PCA, reflecting its sensitivity to distance-preserving transformations.

**Final Takeaways:**
- **Best raw accuracy:** KNN on original data (**92.2%**)
- **Best PCA-reduced accuracy:** Random Forest with 300 estimators (**92.3%**)
- **Best improvement via PCA:** Logistic Regression (**+18.4%** gain)

This project highlights the importance of evaluating models both on raw and transformed features and tailoring preprocessing steps to each model's characteristics.
