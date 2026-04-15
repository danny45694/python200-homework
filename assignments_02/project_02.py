#Week 2 Project
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

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

    plt.hist(dataframe, bins=21)
    plt.title('Distribution of Final Math Grades')
    plt.xlabel("Final period grade")
    plt.ylabel("Scores")

    #Save to outputs folder

    output_file('g3_distribution.png')

hist_distributions(df["G3"])


#Task 2: Preprocess the Data

#Filter out G3 rows first.
df2 = df.drop(df[df["G3"] == 0].index)
#print(df.shape, df2.shape)

df2 = df2.replace({"yes": 1, "no": 0, "F": 0, "M": 1})

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.scatter(df["absences"], df["G3"])
ax1.set_title("Before dropping absent student test scores.")

ax2.scatter(df2["absences"], df2["G3"])
ax2.set_title("After dropping absent student test scores.")

plt.show()

r, p = pearsonr(df['absences'], df['G3'])
r2, p2 = pearsonr(df2['absences'], df2['G3'])
print("Correlation:", round(r, 2))
print("p-value:", round(p, 4))

print('Correlation:', round(r2, 2))
print('p-value:', round(p2, 4))

#Before filtering G3, absences had a correlation of 0, suggesting that there was no relationship between absences and G3 test scores. After filtering, correlation dropped to -0.21. A negative correlation suggests that absences have an somewhat adverse effect on test scores. The correlation is not strong but it plays a factor.

#Task 3: Exploratory Data Analysis

#Double bracket ensures 2D, which sns.heatmap needs
correlation_matrix = df2.corr(method='pearson')[["G3"]]

sorted_mat = correlation_matrix.sort_values(by="G3", ascending=False)
print(sorted_mat)



#seems the Fedu, Medu, age, and sex all have the strongest relationship with G3

#Use annot=True for values
sns.heatmap(sorted_mat, annot=sorted_mat.rank(ascending=False), cmap='coolwarm', center=0)
plt.title("Correlation Rank with G3")
plt.show()


plt.figure(figsize=(10, 6))
sns.barplot(
    data=df2,
    x="Medu",              
    y="G3",                
    hue="Fedu",            
    palette="coolwarm",
)
plt.title("Parent effect on G3 Scores")
plt.xlabel("Mom education level (Medu)")
plt.ylabel("Mean G3 Score")
plt.legend(title="Father's education level (Fedu)")
plt.tight_layout()
plt.show()

# Need to develop 2 charts. Will make a heatmap and hist chart showing distribution of scores.


#Task 4: Baseline Model

X1 = df2[["failures"]]
Y1 = df2[["G3"]]

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

feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime", "G1"]

df_clean = df2.copy()
X = df_clean[feature_cols].values
y = df_clean["G3"].values


x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 
)

T5_model = LinearRegression()
T5_model.fit(x_train, y_train)
y_train_prediction = T5_model.predict(x_train)
y_test_prediction = T5_model.predict(x_test)

t5_rmse = np.sqrt(mean_squared_error(y_test, y_test_prediction))
r2_train = r2_score(y_train, y_train_prediction)
r2_test = r2_score(y_test, y_test_prediction)

for name, coef in zip(feature_cols, T5_model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

#Need to add a comment answering: If I deployed this model in production, which features would you keep and which would you drop? Justify your choices based on what you see in the numbers.

#Task 6: Evaluate and Summarize

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_test_prediction, alpha=0.6, color="steelblue", edgecolors="k", linewidths=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Fit")
plt.title("Actual vs Predicted G3 Scores (Full Model)")
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.legend()
plt.tight_layout()
output_file("actual_vs_predicted.png")


plt.figure(figsize=(10, 5))
coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": T5_model.coef_})
coef_df = coef_df.sort_values("Coefficient", ascending=False)
sns.barplot(data=coef_df, x="Coefficient", y="Feature", palette="coolwarm")
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("Feature Coefficients — Full Model")
plt.tight_layout()
output_file("feature_coefficients.png")

#A high correlation does not equal the root cause. This is definitely a useful model for identifying factors that affect a students performance. If educators wanted to intervene early, providing internet access and study time is the route to go. Educators are unable to control a students background such as family's education levels or economic standing. They are able to control internet and study time however. That also so happens to be a very good indicator (outside of G1) of studies performing well on their final G3 exam. 