import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error, r2_score




# --- scikit-learn API ---

#Q1 


years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_x = np.array([4, 8]).reshape(-1, 1)


#model = ModelClass()    #Create
#model.fit(X_data, y_data)   # Learn from data
#y_predictions = model.predict(X_test)  # Predict on new inputs

"""
model = LinearRegression()
model.fit(years, salary)
y_predicted = model.predict(new_x)
print(y_predicted)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
"""
"""


Code from lesson

from sklearn.linear_model import LinearRegression
import numpy as np

# temperature (degrees C) vs cupcakes sold
x_values = np.array([15, 18, 21, 24, 27]).reshape(-1, 1)
print(x_values.shape, x_values.ndim)
y_values = np.array([150, 200, 240, 310, 400])
new_x = np.array([17, 22]).reshape(-1, 1)

model = LinearRegression()                    # 1. create model
model.fit(x_values, y_values)                 # 2. fit model to data (learn)
y_predicted = model.predict(new_x)            # 3. predict with new data
print(y_predicted)  

"""

#Q2 

#x = np.array([10, 20, 30, 40, 50])
#print(x.shape, x.ndim)

#print(x.reshape(-1,1))

"""
1. We must convert the data to a 2D shape because it provides consistent access across all models, regardless of dataset and features. 
2. 2D array also forces the user to explicitly define the data's structure, ensuring clarity. 
3. Algorithms rely on linear algebra and matrix operations, which are compatible with Numpy 2D array structures and benefit from hardware acceleration.
"""


#Q3 

"""

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
labels = kmeans.predict(X_clusters)

print(kmeans.cluster_centers_)
print(np.bincount(labels))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#Left plot
ax1.scatter(X_clusters[:, 0], X_clusters[:, 1], color='black', s=60, alpha=0.7)
ax1.set_title("Raw Data (No Labels)")
ax1.set_xlabel("xlabel (synthetic scale)")
ax1.set_ylabel("ylabel (synthetic scale)")

ax2.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
ax2.set_title("Data cluster by K-Means")
ax2.set_xlabel("Data clusters (syntheic scale)")

plt.tight_layout()
plt.show()
"""

# ---------------- Linear Regression --------------------

#Q1

#Directory check and file creation functions

#Check if outputs folder exists and create it if false
def create_check_directory():
    output_dir = "outputs"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir

#function to save plot to outputs folder
def output_file(filename):
    output_dir = create_check_directory()
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.show()

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

def question1():
    plt.scatter(age, cost, c=smoker, cmap="coolwarm", alpha=0.5)
    plt.xlabel("Age")
    plt.ylabel('Cost')
    plt.title("Medical Cost vs Age")
    output_file("cost_vs_age.png")
    
#Used to run the function and save file to output folder.
#question1()


#Q2


age = age.reshape(-1, 1)

#after reshape
X_train, X_test, Y_train, Y_test = train_test_split(
    age, cost, test_size=0.2, random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#Q3


model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

#print("Slope:", model.coef_[0])
#print("Intercept:", model.intercept_)


rmse = np.sqrt(np.mean((y_pred - Y_test) ** 2))
r2 = r2_score(Y_test, y_pred)

#print("RMSE:", rmse)
#print("R²:", r2)
# R² presents a score of 0.0695, which means there is little correlation between age and medical costs. The correlation only accounts for 6% of variables. 


#Q4

X_full = np.column_stack([age, smoker])

X_train, X_test, Y_train, Y_test = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)


model_full = LinearRegression()
model_full.fit(X_train, Y_train)
y_pred = model_full.predict(X_test)

#print("Slope:", model_full.coef_[0])
#print("Intercept:", model_full.intercept_)


rmse = np.sqrt(np.mean((y_pred - Y_test) ** 2))
r2 = r2_score(Y_test, y_pred)

#print("age coefficient:    ", model_full.coef_[0])
#print("smoker coefficient: ", model_full.coef_[1])
#print("RMSE:", rmse)
#print("R²:", r2)

# Adding the smoker coefficient shot R² up to 0.7737. This means that being a smoker is much more strongly correlated to paying higher medical costs. 

#Q5

plt.figure(figsize=(8, 6))
plt.scatter(Y_test, y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2) # Diagonal line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs. Predicted (R²: {r2:.2f})')
plt.show()

# This section will contain the comment for Q5 section.See question below


# Add a comment: what does it mean when a point falls above the diagonal? What about below?


# When a point falls above the diagonal, it means the actual y-value is higher than the predicted value. If it falls below, that means the y-value was less than the predicted value. 
