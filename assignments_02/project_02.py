
#Week 2 Project

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error, r2_score


file = "student_performance_math.csv"


#Task 1
df = pd.read_csv(file, sep = ";", )
#print(df.head(5))
#print(df.shape, df.ndim)

def create_check_directory():
    output_dir = "outputs"
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    return output_dir

def output_file(filename):
    output_dir = create_check_directory()
    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.show()




def hist_distributions(dataframe):
    plt.subplot(1,2,1)

    plt.hist(dataframe, bins=21)
    plt.title('Distribution of Final Math Grades')
    plt.xlabel("Final period grade")
    plt.ylabel("Scores")

    #Save to outputs folder

    output_file('g3_distribution.png')

hist_distributions(df["G3"])


#Task 2: Preprocess the Data

df2 = df.drop(df[df["G3"] == 0].index)
#print(df.shape, df2.shape)

df2 = df2.replace({"yes": 1, "no": 0, "F": 0, "M": 1})


r, p = pearsonr(df['absences'], df['G3'])
r2, p2 = pearsonr(df2['absences'], df2['G3'])
#print("Correlation:", round(r, 2))
#print("p-value:", round(p, 4))

#print('Correlation:', round(r2, 2))
#print('p-value:', round(p, 4))

#Create the scatter plots

#plt.scatter(pearson, pearson2)
#Add labels and a title
#plt.show()

#yes/no to 1/0
#sex column to 0/1

#Task 3: Exploratory Data Analysis

correlation_matrix = df2.corr(method='pearson')
G3_comparison = correlation_matrix["G3"]


sorted_mat = G3_comparison.sort_values()
print(sorted_mat)

#seems the Fedu, Medu, age, and sex all have the strongest relationship with G3

# Need to develop 2 charts. Will make a heatmap and hist chart showing distribution of scores.


#Task 4: Baseline Model

X1 = df2["failures"]
Y1 = df2["G3"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X1, Y1, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
r2 = r2_score(Y_test, y_pred)

print("Task 4")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

print("RMSE:", rmse)
print("R²:", r2)

# Postive slope means y increases as x increases. Negative is when the y decreases as x increases. RSME quantifies the average magnitude of errors. A lower RSME means better fit and accuracy. Is R² better or worse than expected.

#Task 5 : Build the Full Model


feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"]

df_clean = pd.DataFrame(feature_cols)
X = df_clean[feature_cols].values
y = df_clean["G3"].values

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 
)

T5_model = LinearRegression()
T5_model.fit(x_train, y_train)
y_prediction = model.predict(x_test)

t5_rmse = np.sqrt(mean_squared_error(y_test, y_prediction))
r2_test = r2_score(y_test, y_prediction)
r2_train = r2_score(y_train, y_prediction)

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

#Need to add a comment answering: If I deployed this model in production, which features would you keep and which would you drop? Justify your choices based on what you see in the numbers.

#Task 6: Evaluate and Summarize



