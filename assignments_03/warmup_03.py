import numpy as np
import matplotlib.pyplot as plt
import os
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

#print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape )


# Q2 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # learns mean and std from training
X_test_scaled = scaler.transform(X_test) # applies the same scaling to test

#print(X_train_scaled)

# If you fit the test data, the model will evaluate better than it should. It's called data leakage. By fitting only the training data and not the test data, you can get a better idea of how new data will perform in real world applications.


#   ------------------------- KNN ---------------------------


#Basic pattern for every model. Applied to most algorithms

"""
model = ModelClass()                      # 1. Create
model.fit(X_data, y_data)                 # 2. Learn from data
y_predictions = model.predict(X_test)     # 3. Predict on new inputs

"""

#Q1 

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)

#print("Accuracy:", accuracy_score(Y_test, predictions))
#print(classification_report(Y_test, predictions ))


#Q2 

knn2 = KNeighborsClassifier(n_neighbors=5)
knn2.fit(X_train_scaled, Y_train)
predictions2 = knn2.predict(X_test_scaled)
#print("Accuracy:", accuracy_score(Y_test, predictions2))
#print(classification_report(Y_test, predictions2 ))

#Q3 

cv_scores = cross_val_score(knn, X_train, Y_train, cv=5)


#print(cv_scores)
#print(f"Mean: {cv_scores.mean(): .3f}")
#print(f"Std: {cv_scores.std():.3f}")


#Using Cross-Validation, the result is more stable. Cross-validation or CV helps ensure that the training partition itself does not skew the results. 

#Q4 

#Need to ask how does one choose which K to choose.

k_values = [1, 3, 5, 7, 9, 11, 13, 15]
cv_scores = []

for k in k_values:

    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, Y_train, cv=5)

    #print(f"k={k:2d}:  mean={scores.mean():.3f}  std={scores.std():.3f}")

#Need to run code above and choose which k I would choose and why.



#----------------------- Classifier Evaluation ---------------------


#Helper functions from assignment_02

def create_check_directory():
    output_dir = "outputs"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir

def output_file(filename):
    output_dir = create_check_directory()
    path = os.path.join(output_dir, filename)
    print (path)
    plt.savefig(path)
    plt.show()



#Q1

## ?? Online tutorials show confusion matrix charts saying machine predict species virginica but the true species is versiscolor. 

## My data has 10, 10, 10. Am I missing something? Says to use predictions from KNN Question 1.

## Running it with prediction2 did not mirror online tutorials either.

def Q1():
    cm = confusion_matrix(Y_test, predictions2)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=iris.target_names
    )
    disp.plot()
    plt.title("KNN Confusion Matrix (Iris)")
    output_file("knn_confusion_matrix.png")

Q1()

# This model is able to classify the data well overall. However, using prediction from Q1 shows perfect predictions. Using the scaled data from Q2, it begins to misclassify the Virginica and Versicolor flowers. AI is telling me this is because the rescales the Sepal Width, which overlaps a lot in different species. Rescaling in this case added noise. ??


#----------------------- Decision Trees ---------------------------
Decision_Tree = DecisionTreeClassifier(max_depth=1)
model.fit()