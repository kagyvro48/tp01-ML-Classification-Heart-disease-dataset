# =============== IMPORTATION DES BIBLIOTHÈQUES ===============
# pandas : pour la manipulation des données
# numpy : pour les calculs numériques
# sklearn : pour les modèles de machine learning et les métriques
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============== CHARGEMENT ET PRÉPARATION DES DONNÉES ===============
# Chargement du dataset depuis GitHub
url_donnees = "https://raw.githubusercontent.com/kagyvro48/tp01-ML-Classification-Heart-disease-dataset/main/heart_classification.csv"
heart_disease = pd.read_csv(url_donnees)

# =============== DIVISION DES DONNEES EN ENTRAINEMENT ET TEST ===============
# X contient toutes les colonnes sauf 'target'
# y contient uniquement la colonne 'target' (0: pas de maladie, 1: maladie)
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

# Division des données en ensembles d'entraînement (80%) et de test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% pour le test
    random_state=42,         # Pour la reproductibilité
    stratify=y               # Pour garder la même distribution de classes
)

# =============== ARBRE DE DÉCISION ===============
# Configuration des hyperparamètres à tester pour l'arbre de décision
params_arbre = {
    'max_depth': [3, 5, 7, 10],          # Profondeur maximale de l'arbre
    'min_samples_split': [2, 5, 10],      # Nombre minimum d'échantillons pour faire un split
    'min_samples_leaf': [1, 2, 4]         # Nombre minimum d'échantillons dans une feuille
}

# CLASSIFICATION HEART DISEASE DATASET
print("\n=== CLASSIFICATION HEART DISEASE DATASET TP01 ===")

# Création du modèle d'arbre avec GridSearch
print("\n=== Entraînement de l'Arbre de Décision ===")
arbre_grille = GridSearchCV(
    DecisionTreeClassifier(random_state=42),  # Le modèle de base
    params_arbre,                             # Les paramètres à tester
    cv=5,                                     # Validation croisée en 5 plis
    scoring='accuracy',                       # Métrique à optimiser
    n_jobs=-1                                # Utiliser tous les processeurs
)

# Entraînement du modèle
arbre_grille.fit(X_train, y_train)

# Affichage des meilleurs paramètres trouvés
print("\nMeilleurs paramètres pour l'arbre :")
print(arbre_grille.best_params_)

# Prédictions sur l'ensemble de test
pred_arbre = arbre_grille.predict(X_test)

# =============== FORÊT ALÉATOIRE ===============
# Configuration des hyperparamètres pour la forêt aléatoire
params_foret = {
    'n_estimators': [100, 200],           # Nombre d'arbres dans la forêt
    'max_depth': [3, 5, 7],              # Profondeur maximale des arbres
    'min_samples_split': [2, 5],         # Échantillons minimums pour split
    'min_samples_leaf': [1, 2]           # Échantillons minimums par feuille
}

# Création du modèle de forêt avec GridSearch
print("\n=== Entraînement de la Forêt Aléatoire ===")
foret_grille = GridSearchCV(
    RandomForestClassifier(random_state=42),  # Le modèle de base
    params_foret,                             # Les paramètres à tester
    cv=5,                                     # Validation croisée en 5 plis
    scoring='accuracy',                       # Métrique à optimiser
    n_jobs=-1                                # Utiliser tous les processeurs
)

# Entraînement du modèle
foret_grille.fit(X_train, y_train)

# Affichage des meilleurs paramètres trouvés
print("\nMeilleurs paramètres pour la forêt :")
print(foret_grille.best_params_)

# Prédictions sur l'ensemble de test
pred_foret = foret_grille.predict(X_test)

# =============== ÉVALUATION DES MODÈLES ===============
# Fonction pour afficher les résultats de manière claire
def afficher_resultats_classification(y_vrai, y_pred, nom_modele):
    print(f"\n=== Résultats pour {nom_modele} ===")
    print(f"Précision (Accuracy) : {accuracy_score(y_vrai, y_pred):.3f}")
    print("\nRapport de classification détaillé :")
    print(classification_report(y_vrai, y_pred))
    print("\nMatrice de confusion :")
    print(confusion_matrix(y_vrai, y_pred))

# Affichage des résultats pour les deux modèles
afficher_resultats_classification(y_test, pred_arbre, "l'Arbre de Décision")
afficher_resultats_classification(y_test, pred_foret, "la Forêt Aléatoire")

# =============== IMPORTANCE DES FEATURES ===============
# Récupération et affichage de l'importance des features pour la forêt aléatoire
importance_features = pd.DataFrame({
    'Feature': X.columns,
    'Importance': foret_grille.best_estimator_.feature_importances_
})
importance_features = importance_features.sort_values('Importance', ascending=False)

print("\n=== Importance des caractéristiques (Forêt Aléatoire) ===")
print(importance_features)
