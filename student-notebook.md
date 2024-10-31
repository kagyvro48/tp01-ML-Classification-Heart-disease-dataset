# TP01 - Apprentissage Supervisé
## Classification et Régression avec Arbres de Décision et Forêts Aléatoires

### Introduction
Dans ce TP, on va travailler sur deux problèmes différents :
1. Classification pour prédire les maladies cardiaques
2. Régression pour prédire les prix des maisons en Californie

Je vais expliquer étape par étape ce qu'on fait et pourquoi on le fait.

### Partie 1 : Classification sur Heart Disease Dataset

Premièrement, on importe les bibliothèques dont on a besoin :

```python
# On importe les bibliothèques nécessaires
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

Maintenant, on charge les données :

```python
# Chargement du dataset depuis GitHub
url_donnees = "https://raw.githubusercontent.com/kagyvro48/tp01-ML-Classification-Heart-disease-dataset/main/heart_classification.csv"
heart_disease = pd.read_csv(url_donnees)
```

#### Préparation des données

On doit séparer nos features (X) de notre target (y) :

```python
# X contient toutes les colonnes sauf 'target'
# y contient uniquement la colonne 'target' (0: pas de maladie, 1: maladie)
X = heart_disease.drop('target', axis=1)
y = heart_disease['target']

# On divise en train (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% pour le test
    random_state=42,  # Pour avoir les mêmes résultats à chaque fois
    stratify=y  # Pour garder la même proportion de chaque classe
)
```

#### Entraînement des modèles

##### 1. Arbre de décision
On va d'abord essayer différentes valeurs de paramètres pour trouver les meilleurs :

```python
# Configuration des paramètres à tester
params_arbre = {
    'max_depth': [3, 5, 7, 10],  # Profondeur max de l'arbre
    'min_samples_split': [2, 5, 10],  # Échantillons min pour faire un split
    'min_samples_leaf': [1, 2, 4]  # Échantillons min dans une feuille
}

# Création et entraînement du modèle avec GridSearch
arbre_grille = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    params_arbre,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

arbre_grille.fit(X_train, y_train)
```

Les meilleurs paramètres trouvés sont :
- max_depth: 10
- min_samples_leaf: 1
- min_samples_split: 2

##### 2. Forêt aléatoire
Même chose pour la forêt aléatoire :

```python
# Configuration des paramètres
params_foret = {
    'n_estimators': [100, 200],  # Nombre d'arbres
    'max_depth': [3, 5, 7],  # Profondeur max
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

foret_grille = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params_foret,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

foret_grille.fit(X_train, y_train)
```

Les meilleurs paramètres trouvés sont :
- n_estimators: 100
- max_depth: 7
- min_samples_leaf: 1
- min_samples_split: 2

#### Résultats de la classification

Pour l'arbre de décision :
- Précision (Accuracy) : 0.985 (98.5%)
- La matrice de confusion montre :
  - 100 vrais négatifs (pas de maladie correctement prédit)
  - 102 vrais positifs (maladie correctement prédite)
  - 3 faux négatifs (maladie non détectée)
  - 0 faux positifs (pas de fausse alerte)

Pour la forêt aléatoire :
- Précision (Accuracy) : 0.990 (99%)
- La matrice de confusion montre :
  - 98 vrais négatifs
  - 105 vrais positifs
  - 0 faux négatifs (super important pour un diagnostic !)
  - 2 faux positifs

Les caractéristiques les plus importantes d'après la forêt aléatoire sont :
1. cp (douleur thoracique) : 14.53%
2. ca (nombre de vaisseaux majeurs) : 12.39%
3. thalach (fréquence cardiaque max) : 10.83%

### Partie 2 : Régression sur California Housing Dataset

On utilise les mêmes imports que précédemment, plus quelques nouveaux :

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
```

#### Préparation des données

```python
# Chargement du dataset
url_donnees = "https://raw.githubusercontent.com/kagyvro48/tp01-ML-Regression-California-Houses-Prices-dataset/main/housing.csv"
california_housing = pd.read_csv(url_donnees)

# On encode la colonne ocean_proximity en nombres
label_encoder = LabelEncoder()
california_housing['ocean_proximity_encoded'] = label_encoder.fit_transform(california_housing['ocean_proximity'])

# Préparation de X et y
X = california_housing.drop(['median_house_value', 'ocean_proximity'], axis=1)
y = california_housing['median_house_value']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

#### Entraînement des modèles

Les paramètres testés et l'entraînement sont similaires à la classification, mais avec des régresseurs au lieu de classificateurs.

#### Résultats de la régression

Pour l'arbre de décision :
- RMSE : 60,798.52 dollars
- MAE : 39,736.55 dollars
- R² : 0.718 (71.8% de la variance expliquée)

Pour la forêt aléatoire :
- RMSE : 50,165.52 dollars
- MAE : 32,328.95 dollars
- R² : 0.808 (80.8% de la variance expliquée)

Les caractéristiques les plus importantes sont :
1. median_income (53.78%)
2. ocean_proximity_encoded (11.41%)
3. latitude (10.70%)
4. longitude (10.44%)

### Discussion et interprétation

1. Comparaison des performances :
   - Pour la classification : La forêt aléatoire (99%) est légèrement meilleure que l'arbre de décision (98.5%)
   - Pour la régression : La forêt aléatoire est clairement meilleure (R² de 0.808 vs 0.718)

2. Pourquoi la forêt aléatoire est meilleure ?
   - Elle utilise plusieurs arbres (100 ou 200) au lieu d'un seul
   - Chaque arbre voit des données un peu différentes
   - La décision finale est prise en groupe (comme plusieurs médecins qui donnent leur avis)

3. Points importants :
   - Pour les maladies cardiaques : les symptômes physiques (douleur thoracique, vaisseaux) sont les plus importants
   - Pour les prix des maisons : le revenu médian est super important (plus de 50% !)
   - La forêt aléatoire fait moins d'erreurs graves dans les deux cas

4. Limites et améliorations possibles :
   - On pourrait essayer plus de valeurs de paramètres
   - Pour les maisons, on pourrait ajouter des features (âge du quartier, proximité des écoles...)
   - Les données ne sont peut-être pas très récentes

### Conclusion

Les deux modèles marchent bien, mais la forêt aléatoire est généralement meilleure. Pour un diagnostic médical, avoir moins de faux négatifs (cas de la forêt aléatoire) est très important. Pour les prix des maisons, la forêt aléatoire fait des erreurs moins importantes en moyenne.

Je pense qu'on devrait utiliser la forêt aléatoire dans les deux cas, même si elle prend un peu plus de temps à s'entraîner.
