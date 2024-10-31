
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
[Finished in 110.9s]