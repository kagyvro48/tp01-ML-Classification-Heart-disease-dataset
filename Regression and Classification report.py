Python 3.12.6 (tags/v3.12.6:a4a2d2b, Sep  6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.

= RESTART: C:\Users\KAGYVRO\Desktop\Code Python\Classification-Heart-disease-dataset-without-Graphs.py

=== CLASSIFICATION HEART DISEASE DATASET TP01 ===

=== Entraînement de l'Arbre de Décision ===

Meilleurs paramètres pour l'arbre :
{'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}

=== Entraînement de la Forêt Aléatoire ===

Meilleurs paramètres pour la forêt :
{'max_depth': 7, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100}

=== Résultats pour l'Arbre de Décision ===
Précision (Accuracy) : 0.985

Rapport de classification détaillé :
              precision    recall  f1-score   support

           0       0.97      1.00      0.99       100
           1       1.00      0.97      0.99       105

    accuracy                           0.99       205
   macro avg       0.99      0.99      0.99       205
weighted avg       0.99      0.99      0.99       205


Matrice de confusion :
[[100   0]
 [  3 102]]

=== Résultats pour la Forêt Aléatoire ===
Précision (Accuracy) : 0.990

Rapport de classification détaillé :
              precision    recall  f1-score   support

           0       1.00      0.98      0.99       100
           1       0.98      1.00      0.99       105

    accuracy                           0.99       205
   macro avg       0.99      0.99      0.99       205
weighted avg       0.99      0.99      0.99       205


Matrice de confusion :
[[ 98   2]
 [  0 105]]

=== Importance des caractéristiques (Forêt Aléatoire) ===
     Feature  Importance
2         cp    0.145340
11        ca    0.123912
7    thalach    0.108256
12      thal    0.106809
9    oldpeak    0.103420
0        age    0.090238
8      exang    0.083978
4       chol    0.067971
3   trestbps    0.064271
10     slope    0.052095
1        sex    0.027167
6    restecg    0.016421
5        fbs    0.010121
>>> 
= RESTART: C:\Users\KAGYVRO\Desktop\Code Python\TP01-ML-Regression-California-houses-prices-dataset.py

=== REGRESSION CALIFORNIA HOUSES PRICES DATASET TP01 ===

=== Entraînement de l'Arbre de Décision ===

Meilleurs paramètres pour l'arbre :
{'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5}

=== Entraînement de la Forêt Aléatoire ===

Meilleurs paramètres pour la forêt :
{'max_depth': 15, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 200}

=== Résultats pour l'Arbre de Décision ===
Erreur quadratique moyenne (MSE) : 3,696,459,815.69
Racine de l'erreur quadratique moyenne (RMSE) : 60,798.52
Erreur absolue moyenne (MAE) : 39,736.55
Coefficient de détermination (R²) : 0.718

=== Résultats pour la Forêt Aléatoire ===
Erreur quadratique moyenne (MSE) : 2,516,579,711.06
Racine de l'erreur quadratique moyenne (RMSE) : 50,165.52
Erreur absolue moyenne (MAE) : 32,328.95
Coefficient de détermination (R²) : 0.808

=== Importance des caractéristiques (Forêt Aléatoire) ===
                   Feature  Importance
7            median_income    0.537840
8  ocean_proximity_encoded    0.114057
1                 latitude    0.107024
0                longitude    0.104426
2       housing_median_age    0.052139
5               population    0.029443
4           total_bedrooms    0.020492
3              total_rooms    0.020158
6               households    0.014421
