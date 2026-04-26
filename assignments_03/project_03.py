import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)



warnings.filterwarnings("ignore", category=RuntimeWarning)

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
#print(df.head())

features = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']
"""
for feature in features:
    spam = df[df['spam_label'] == 1][feature]
    ham = df[df['spam_label'] == 0][feature]
    
    plt.figure()
    plt.boxplot([ham, spam], tick_labels=['Ham', 'Spam'])
    plt.title(f'{feature} by class')
    plt.ylabel(feature)
    plt.savefig(f'outputs/{feature}_boxplot.png')
    plt.show()
"""
# The differences between classes are more subtle. Emails using the word "free" are more likely to be spam. The emails attempt to entice the reader into taking action with the promise of no cost. 

#Need to double check comment here. Not sure if I understood question correctly.


# Task 2


X = df.drop(columns=['spam_label']) # Grab all 57 feature columns
y = df['spam_label'] #Target

#Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Scale

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # learns means and standard deviation from training
X_test_scaled = scaler.transform(X_test)


pca = PCA()
pca.fit(X_train_scaled)

#Cumulative_variance - tells how many components will account for a certain percentage of the data

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.savefig('outputs/UCI_variance.png')
#plt.show()

#Once you have the num of N components. Use code below

#Looks like number of N components that accounts for 90% is 45.

n = 45

# argmax can doublecheck your n component variable
n_doublecheck = np.argmax(cumulative_variance >= 0.90) + 1
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]


# Task 3

#Unscaled data
knn_unscaled = KNeighborsClassifier(n_neighbors= 5)
knn_unscaled.fit(X_train, y_train)
predictions_unscaled = knn_unscaled.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions_unscaled))
print(classification_report(y_test, predictions_unscaled ))



#Scaled

"""
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
prediction_scaled = knn_scaled.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, prediction_scaled))
print(classification_report(y_test, prediction_scaled ))
"""


#PCA-reduced data

"""
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
predictions_pca = knn_pca.predict(X_test_pca)
print("Accuracy:", accuracy_score(y_test, predictions_pca))
print(classification_report(Y_test, predictions_pca))
"""


#Decision-Tree

"""

max_depth = [3,5,10, None]

for i in max_depth:
    Decision_Tree = DecisionTreeClassifier(max_depth=i, random_state=42)
    decision_fit = Decision_Tree.fit(X_train, y_train)
    decision_prediction = Decision_Tree.predict(X_test)
    importance = pd.Series(Decision_Tree.feature_importances_, index=X_train.columns)
    train_accuracy = accuracy_score(y_train, Decision_Tree.predict(X_train))
    test_accuracy = accuracy_score(y_test, decision_prediction)
    print(importance.nlargest(10))
    print(f"max_depth={i}: train={train_acc:.3f} test={test_acc:.3f}")



print("Accuracy:", accuracy_score(y_test, decision_prediction))
print(classification_report(y_test, decision_prediction ))
"""


#Random Forest Classifer
"""
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_,index=X_train.columns)
#Create a bar chart of the Random Forest importance to (outputs/feature_importances.png)

importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Random Forest Feature Importances")
plt.tight_layout()
plt.savefig('outputs/feature_importances.png')
plt.show()


#Logistic Regression

model = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
model.fit(X_train_scaled, y_train)
total = np.abs(model.coef_).sum()
print(total)

model2 = LogisticRegression(C=1.0, max_iter=1000, solver ='liblinear')
model2.fit(X_train_pca, y_train)
total2 = np.abs(model2.coef_)
print(total2)

"""


#Task 4

"""
#Confusion matrix for the best performing classifier
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)
disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
output_file("knn_confusion_matrix.png")}
"""