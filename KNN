import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy (k={k}): {acc:.2f}")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(knn, scaler.transform(X), y, cv=5, scoring='accuracy')
print("\nCross-validation accuracies:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

param_grid = {'n_neighbors': np.arange(1, 21)}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(scaler.transform(X), y)
best_k = grid.best_params_['n_neighbors']
print(f"\nBest k found using GridSearchCV: {best_k}")
print(f"Best Cross-validation accuracy: {grid.best_score_:.3f}")

cv_means = []
for k in range(1, 21):
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                             scaler.transform(X), y, cv=5, scoring='accuracy')
    cv_means.append(scores.mean())

plt.figure(figsize=(8,4))
plt.plot(range(1, 21), cv_means, marker='o', linestyle='-')
plt.title("K Value vs Cross-Validation Accuracy")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("CV Accuracy")
plt.grid(True)
plt.show()

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
final_pred = final_knn.predict(X_test_scaled)
final_acc = accuracy_score(y_test, final_pred)
print(f"\nFinal Model Accuracy (k={best_k}): {final_acc:.2f}")
