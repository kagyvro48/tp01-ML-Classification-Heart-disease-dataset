from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Charger le dataset Iris
iris = load_iris()
X = iris.data
y = iris.target

# Diviser le dataset en training et testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle K-Nearest Neighbors (KNN)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Faire des prédictions
y_pred = model.predict(X_test)

# Calculer l'accuracy du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy du modèle: {accuracy * 100:.2f}%")
print(y_pred)
print(y_test)
