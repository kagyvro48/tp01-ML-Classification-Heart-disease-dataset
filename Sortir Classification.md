
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
[Finished in 10.7s]