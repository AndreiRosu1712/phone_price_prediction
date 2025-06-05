# Clasificarea Gamelor de Prețuri pe Bază de Caracteristici

Acest proiect folosește tehnici de învățare automată pentru a clasifica gama de preț (`price_range`) a unui produs pe baza unui set de caracteristici numerice și categorice. Sunt aplicate mai multe modele de clasificare și se compară performanțele acestora atât pe datele originale, cât și pe date reduse cu PCA.

## Date de intrare

Fișierul principal de date este:
- `trainPrice.csv`: setul de antrenament care conține caracteristici și eticheta țintă `price_range`.

## Biblioteci necesare

- pandas
- numpy
- seaborn
- matplotlib
- sklearn (model_selection, metrics, decomposition, linear_model, ensemble, neighbors)

## Explorarea datelor

- Încărcarea setului de date și afișarea informațiilor generale (`shape`, `head`, `info`, `describe`, `isnull().sum()`).
- Separarea caracteristicilor în:
  - Caracteristici categorice: cu mai puțin de 30 valori unice.
  - Caracteristici numerice: cu 30 sau mai multe valori unice.
- Vizualizarea distribuției variabilelor:
  - Ploturi de tip countplot pentru variabilele categorice.
  - Histogramă și KDE pentru variabilele numerice, împreună cu media marcată.

## Preprocesare

- Variabilă țintă: `price_range`
- Caracteristici: toate celelalte coloane
- Împărțirea setului de date în:
  - `X_train`, `X_test`, `y_train`, `y_test` (75% / 25%)

## Reducerea dimensionalității

Se aplică PCA (Principal Component Analysis) pentru reducerea dimensionalității la un prag de 95% varianță explicată:
- `X_train_reduced`
- `X_test_reduced`

## Modele antrenate

### Logistic Regression

- Aplicat pe datele originale
- Aplicat pe datele reduse cu PCA
- Se calculează acuratețea și se afișează matricea de confuzie

### Random Forest

- Aplicat pe datele originale
- Aplicat pe datele reduse cu PCA
- Se evaluează mai multe valori pentru `n_estimators` (10, 50, 100, 200, 300) folosind `cross_val_score`
- Se calculează acuratețea și se afișează matricea de confuzie

### K-Nearest Neighbors (KNN)

- `n_neighbors=5`
- Aplicat pe datele originale
- Aplicat pe datele reduse cu PCA
- Se calculează acuratețea și se afișează matricea de confuzie

## Evaluare

Pentru fiecare model sunt calculate:
- Acuratețea (`accuracy_score`)
- Matricea de confuzie (`ConfusionMatrixDisplay`)

## Observații și posibile erori

- Funcția `confusion_matrix` nu este importată direct (ar trebui: `from sklearn.metrics import confusion_matrix`).
- Variabilele `knnoriginal` și `knnreduced` nu sunt definite corect în afișarea matricei de confuzie (`display_labels`); probabil ar trebui înlocuite cu `knn_original` și `knn_reduced`.

## Recomandări

- Adăugarea unui pipeline de preprocesare pentru scalarea caracteristicilor.
- Normalizarea datelor înainte de antrenarea KNN.
- Salvarea rezultatelor și graficelor într-un folder de ieșire.
- Corectarea typo-urilor și importurilor lipsă pentru o execuție completă.

## Concluzie

Scriptul compară performanțele a trei modele de clasificare (Logistic Regression, Random Forest, KNN) pe seturi de date originale și reduse, analizând impactul reducerii dimensionalității asupra preciziei modelelor.
