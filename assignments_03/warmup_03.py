import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


# ------------------------ Preprocessing ----------------------------

 
 
# Q1


X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape )


# Q2 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # learns mean and std from training
X_test_scaled = scaler.transform(X_test) # applies the same scaling to test

print(X_train_scaled)

# If you fit the test data, the model will evaluate better than it should. It's called data leakage. By fitting only the training data and not the test data, you can get a better idea of how new data will perform in real world applications.


#   ------------------------- KNN ---------------------------


#Q1 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)

preds = knn.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, preds))
print(classification_report(Y_test, preds ))


#Q2 

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_scaled, Y_train)
preds = knn2.predict(X_test_scaled)

print("Accuracy:", accuracy_score(Y_test, preds))
print(classification_report(Y_test, preds ))

#Q3 

cv_scores = cross_val_score(knn, X_train, Y_train, cv=5)

print(cv_scores)
print(f"Mean: {cv_scores.mean(): .3f}")
print(f"Std: {cv_scores.std():.3f}")

#Is the result more or less trustworthy than a single train/test split? Why?

#Q4 

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
cv_scores = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=5)
    mean_score = np.mean(scores)
    cv_scores.append(mean_score)

    print(f"k = {k}, Mean CV Accuracy = {mean_score}")

#Need to run code above and choose which k I would choose and why.