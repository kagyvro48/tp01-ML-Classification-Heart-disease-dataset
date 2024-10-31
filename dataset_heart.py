# Importer les bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Charger les données de classification
url = "https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset"
data = pd.read_csv(url)

# Prétraitement des données
X = data.drop(columns=['target'])  # Variables indépendantes
y = data['target']  # Variable cible

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de classification basé sur l'arbre de décision
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# Évaluer l'arbre de décision
print("Accuracy de l'arbre de décision:", accuracy_score(y_test, y_pred_tree))
print(classification_report(y_test, y_pred_tree))

# Modèle de classification basé sur les forêts aléatoires
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)

# Évaluer la forêt aléatoire
print("Accuracy de la forêt aléatoire:", accuracy_score(y_test, y_pred_forest))
print(classification_report(y_test, y_pred_forest))

# Utiliser Grid Search pour optimiser les hyperparamètres
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Meilleurs hyperparamètres:", grid_search.best_params_)
