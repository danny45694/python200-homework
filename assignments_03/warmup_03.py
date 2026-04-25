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
from sklearn.multiclass import OneVsRestClassifier
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

#Q1()

# This model is able to classify the data well overall. However, using prediction from Q1 shows perfect predictions. Using the scaled data from Q2, it begins to misclassify the Virginica and Versicolor flowers. AI is telling me this is because the rescales the Sepal Width, which overlaps a lot in different species. Rescaling in this case added noise. ??


#----------------------- Decision Trees ---------------------------

"""
Decision_Tree = DecisionTreeClassifier(max_depth=3, random_state=42)
decision_fit = (X_train, Y_train)
decision_prediction = Decision_Tree.predict(X_test)


print("Accuracy:", accuracy_score(Y_test, decision_prediction))
print(classification_report(Y_test, decision_prediction ))


"""

# Distance-related algorithms have no impact on the accuracy or structure of decision trees. So answer is no.


#----------------Logistic Regression and Regularization------------

"""
Global variables. Brought here for reference.

X_train_scaled = scaler.fit_transform(X_train) # learns mean and std from training
X_test_scaled = scaler.transform(X_test) # applies the same scaling to test
"""

# Train Logistic Regression Model with 1 feature


OneVsRestClassifier(LogisticRegression(C=0.01, max_iter=1000, solver="liblinear")),
OneVsRestClassifier(LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")),
OneVsRestClassifier(LogisticRegression(C=100, max_iter=1000, solver="liblinear"))





c_values = [0.01, 1.0, 100]

for c in c_values:
    model = LogisticRegression(C=c, max_iter=1000, solver='liblinear')
    model.fit(X_train_scaled, Y_train)
    total = np.abs(model.coef_).sum()
    print(f"C={c}: total coefficient magnitude = {total: .4f}")


#-------------------------------PCA--------------------------------
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

#Q1

#print(X_digits.shape, images.shape)

"""
for digit in range(10):
    index = list(y_digits).index(digit)

    plt.subplot(1,10, digit + 1)
    plt.imshow(images[index], cmap='gray_r')
    plt.title(digit)
    plt.xlabel
    output_file("sample_digits.png")
"""


"""
#Q2


pca = PCA()
pca_fit = pca.fit(X_digits)
scores = pca.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10) # c = color array
plt.colorbar(scatter, label="Digit")
plt.savefig('outputs/pca_2d_projection.png')
#plt.show()

#Yes they do. Images tend to cluster together in the same area. Some have values that overlap with other images but it tends to have a clear section for the most part. 


#Q3

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.savefig('outputs/pca_variance_explained.png')
#plt.show()

#We need around 20 components to explain 80% of the variance


#Q4

def reconstruct_digit(sample_idx, scores, pca, n_components):
    #Reconstruct one digit using the first n_components principal components.
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

components = pca.components_

# Use the function below to generate PCA question 4
#reconstruct_digit(samples_idx, scores,pca, n_components)

#scores = pca.transform(X_digits)

fig, axes = plt.subplots(5, 5)
for column in range(5):
    axes[0, column].imshow(images[column], cmap="gray_r")
    axes[0, column].set_title(f"Digit {y_digits[column]}")
    axes[0, column].axis("off")

axes[0, 0].set_ylabel("Original")

n_values = [2, 5, 15, 40]
for row, n in enumerate(n_values, start=1):
    for column in range(5):
        reconstructed = reconstruct_digit(column, scores, pca, n)
        axes[row, column].imshow(reconstructed, cmap="gray_r")
        axes[row, column].axis("off")

    axes[row, 0].set_ylabel(f"n={n}")
    plt.savefig("outputs/pca_reconstructions.png")

plt.show()

# At n = 15 is when the digits become recognizable. It definitely correlates to where the variance curve levels off.

"""