# =============== IMPORTATION DES BIBLIOTHÈQUES ===============
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =============== CHARGEMENT ET PRÉPARATION DES DONNÉES ===============
# Chargement du dataset
url_donnees = "https://raw.githubusercontent.com/kagyvro48/tp01-ML-Regression-California-Houses-Prices-dataset/main/housing.csv"
california_housing = pd.read_csv(url_donnees)

# =============== PRÉTRAITEMENT DES DONNÉES ===============
# Séparation des caractéristiques numériques et catégorielles
colonnes_numeriques = california_housing.select_dtypes(include=['int64', 'float64']).columns.tolist()
colonnes_numeriques.remove('median_house_value')  # Retirer la variable cible

# Créer des pipelines de prétraitement
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Encoder la variable catégorielle ocean_proximity
label_encoder = LabelEncoder()
california_housing['ocean_proximity_encoded'] = label_encoder.fit_transform(california_housing['ocean_proximity'])

# Préparer X et y
X = california_housing.drop(['median_house_value', 'ocean_proximity'], axis=1)
y = california_housing['median_house_value']

# Division des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =============== ARBRE DE DÉCISION ===============
# Configuration des hyperparamètres
params_arbre = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# REGRESSION CALIFORNIA HOUSES PRICES DATASET
print("\n=== REGRESSION CALIFORNIA HOUSES PRICES DATASET TP01 ===")

# Création et entraînement du modèle
print("\n=== Entraînement de l'Arbre de Décision ===")
arbre_grille = GridSearchCV(
    DecisionTreeRegressor(random_state=42),
    params_arbre,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

arbre_grille.fit(X_train, y_train)

print("\nMeilleurs paramètres pour l'arbre :")
print(arbre_grille.best_params_)

# Prédictions
pred_arbre = arbre_grille.predict(X_test)

# =============== FORÊT ALÉATOIRE ===============
# Configuration des hyperparamètres
params_foret = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Création et entraînement du modèle
print("\n=== Entraînement de la Forêt Aléatoire ===")
foret_grille = GridSearchCV(
    RandomForestRegressor(random_state=42),
    params_foret,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

foret_grille.fit(X_train, y_train)

print("\nMeilleurs paramètres pour la forêt :")
print(foret_grille.best_params_)

# Prédictions
pred_foret = foret_grille.predict(X_test)

# =============== ÉVALUATION DES MODÈLES ===============
def afficher_resultats_regression(y_vrai, y_pred, nom_modele):
    print(f"\n=== Résultats pour {nom_modele} ===")
    mse = mean_squared_error(y_vrai, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_vrai, y_pred)
    r2 = r2_score(y_vrai, y_pred)
    
    print(f"Erreur quadratique moyenne (MSE) : {mse:,.2f}")
    print(f"Racine de l'erreur quadratique moyenne (RMSE) : {rmse:,.2f}")
    print(f"Erreur absolue moyenne (MAE) : {mae:,.2f}")
    print(f"Coefficient de détermination (R²) : {r2:.3f}")

# Affichage des résultats pour les deux modèles
afficher_resultats_regression(y_test, pred_arbre, "l'Arbre de Décision")
afficher_resultats_regression(y_test, pred_foret, "la Forêt Aléatoire")

# =============== IMPORTANCE DES FEATURES ===============
importance_features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': foret_grille.best_estimator_.feature_importances_
})
importance_features = importance_features.sort_values('Importance', ascending=False)

print("\n=== Importance des caractéristiques (Forêt Aléatoire) ===")
print(importance_features)
