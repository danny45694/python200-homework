import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from scipy import stats
import seaborn as sns


#!! --- I placed all question prints as a comment to declutter the terminal output section. 

#Q1
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

#print(f"Num Rows: {len(df)}")

#Q2 Filter dataframe to only show students who passed and have scores above 80
students = df[df["grade"] > 80]
#print(students)

#Q3 Add new column "Grade_curved", add 5 points to each student's grade. Print updated dataframe

df["grade_curved"] = df["grade"] + 5
#print(df)

#Q4 Add new column "name_upper" that contains each student's name in uppercase using .str accessor. Print "name" and name_upper columns together

df["name_upper"] = df["name"].str.upper()
#print(df["name_upper"], df["name"])

#Q5 Group by city and compute mean grade for each city. Print result

#print(df.groupby("city")['grade'].mean())

#Q6 Replace Austin in "city" with "Houston". Print name and city columns

df["city"] = df['city'].str.replace("Austin", "Houston")

#print(df["name"], df["city"])

#Q7 Sort dataframe by "grade" in descending order. Print top 3 rows

sorted_by_grade = df.sort_values(by="grade", ascending=False)
#print(sorted_by_grade.head(3))


#----------------------NUMPY---------------------

#Q1 Create 1D NumPy array from list [10,20,30,40,50]. Print shape, dtype and ndim

a = np.array([10,20,30,40,50])
#print(a.shape, a.dtype, a.ndim)

#Q2 Create the following 2D array and print its shape and size

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

#print(arr.shape, arr.size)

#Q3 Using a 2D array from Q2, slice out the top-left 2x2 block and print it. Expected result is [[1,2], [4,5]]


#print(arr[0:2, 0:2])


#Q4 Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both

zero = np.zeros((3,4))
#print(zero)

ones = np.ones((2,5))
#print(ones)

#Q5 Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like. Then, print the array, its shape, mean, sum, and standard deviation.

arange = np.arange(0, 50, 5)
#Array will be a 1D array with numbers going in sequence of 5.

#print(arange, arange.shape, arange.mean(), arange.sum(), arange.std())

#Q6 Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.

mean, devi = 0, 1 # mean and standard deviation

array = np.random.normal(mean, devi, 200)

#print(array, array.std(), array.mean())


#----------------------------------MATPLOTLIB REVIEW-----------------------------


#Q1 Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".

#Prepared data. (NumPy arrays usually used)
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

#Plot customization (Optional but recommended)
"""
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Squares")
"""
#Display Plot
#plt.show()

#Q2 Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.

#Data
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]



#Basic syntax for pyplot is plt.[type](x,y)
#   It can be plt.plot or plt.bar or plt.scatter. See written notes
"""
#Edgecolor colors the bar edges. linewidth draws the width of the bar edge(s)
plt.bar(subjects, scores, color= 'blue', edgecolor='white', linewidth=0.7)
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.title("Subject Scores")
plt.show()
"""




#Q3 Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.
"""
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1,y1,label = "Dataset 1", color="blue")
plt.scatter(x2,y2,label="Dataset 2", color="red", marker='s' )
plt.legend()
plt.show()

"""

#Q4 Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing.


"""
#Prepared data from Q1. 
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

#Plot 1 (left side plot)

plt.subplot(1, 2, 1) # 1 row, 2 columns, index 1
plt.plot(x, y, label="Q1", color="green")
plt.xlabel("x")
plt.ylabel('y')
plt.title("Q1")
plt.legend()

#Data from Q2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.subplot(1, 2, 2) #1 row, 2 columns, index 2
plt.bar(subjects, scores, label="Q2", color="blue", edgecolor="white", linewidth= 0.7)
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.title("Subject Scores")
plt.legend()


#Optional: Adjust layout so titles don't overlap
plt.tight_layout()
plt.show()
"""
#----------------------Descriptive Stats review---------------------



#Q1 Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. Label each printed value.

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

#Creating NumPy array
data_np = np.array([data])
median = np.median(data_np)


#print(median, data_np.std(), data_np.mean(), data_np.var())


#Q2 Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
"""
q2data = np.random.normal(65, 10, 500)

plt.hist(q2data, bins=20, color='skyblue', edgecolor='black')

plt.xlabel('Standard Deviation')
plt.ylabel('Scores')
plt.title('Distribution of Scores')
plt.show()
"""

#Q3 Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
group = [group_a, group_b]

#Hint: labels=["Group A", "Group B"]

"""
plt.boxplot(group, tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.show()
"""


#Q4 You are given two datasets: one normally distributed and one 'exponential' distribution. See instructions and code below

"""
Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison".

Then, add a comment in your code briefly noting which distribution is more skewed, and which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution.

"""

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)


"""
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Value")
plt.show()

"""

#Need to add a comment on which descriptive statistic (mean or median) would be more appropriate measure of central tendency for each distribution

#Q5 Print the mean, median, and mode of the following:

#Additional prompt: Why are the median and mean so different for data2? Add your answer as a comment in the code.

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

#print("Data 1:", "Mean:", np.mean(data1), "Median:",  np.median(data1), "Mode:", stats.mode(data1), sep="\n")
#print("Data 2:", "Mean:", np.mean(data2), "Median:",  np.median(data2), "Mode:", stats.mode(data2), sep="\n")



#Prompt answer
"""
    Mean returns value total/ num of values
    
    Median returns the middle value when data is sorted. It takes the avg. of the 2 middle values when a dataset has an even number of values

    Mode returns the most frequently occuring value. Not useful for continuous numeric data(like heights) because exact repeats are rare.

    The mean is different because data2 contains 150. The total sum / num of values is larger in data2. The Median remains the same for me. Equal number of items in the list.

"""


#--------------------------------- Hypothesis testing ---------------------------------

#Q1 Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.



group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]


t_stat, p_val = stats.ttest_ind(group_a, group_b)

#print("t-statistic:", t_stat)
#print("p-value:", p_val)



#Q2 Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05

"""
if p_val < 0.05:
    print("The difference is statistically significant")
else:
    print("No statistically significant difference detected")
"""

#Q3 Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_statistic, p_value = stats.ttest_rel(before, after)

#print(t_statistic, p_value)

#Q4 Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value.

scores = [72, 68, 75, 70, 69, 74, 71, 73]

#q4_tstat, p_val = stats.test_1samp(scores, 70)
#print(q4_tstat, p_val)

#Q5 Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter.

#p_value = stats.ttest_ind(group_a, group_b, alternative="less")
#print(p_value)

#Q6 Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis"). Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.

"""
print("The difference in average scores is unlikely due to random chance. ")
"""

#---------------------------- Correlation Review -----------------------------

#Q1 Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1]).

"""
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]


corr_matrix = np.corrcoef(x, y)
print(corr_matrix)
"""

#Q2 Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.


"""
from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

r, p = pearsonr(x, y)
print("Correlation:", round(r, 2))
print("p-value:", round(p, 4))
"""


#Q3 Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result

"""
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

print(df.corr())
"""


#Q4 Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.


"""

x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]


plt.scatter(x, y, color='teal')
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#Q5 Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap"

sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

"""



#-------------------------- Pipelines -----------------------------


#Q1 A data pipeline is a sequence of processing steps where each step takes in data, transforms it, and passes the result to the next. You don't need a special framework to build one -- chaining plain functions together is often enough. 

# Given the array below, which contains some missing values scattered throughout:


arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    values = pd.Series(arr)
    return values

def clean_data(series):
    values = series.dropna()
    return values

def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary


def data_pipeline(arr):
    values = create_series(arr)
    clean_data = clean_data(values)
    summary = summarize_data(values)
    return summary


print(data_pipeline(arr))

"""
Implement the following three functions and then connect them in a data_pipeline() function.

create_series(arr) : takes a NumPy array and returns a pandas Series with the name "values".
clean_data(series) : takes the Series, removes any NaN values using .dropna(), and returns the cleaned Series.
summarize_data(series) -- takes the cleaned Series and returns a dictionary with four keys: "mean", "median", "std", and "mode". For mode, use series.mode()[0] to get a single value.
data_pipeline(arr) -- calls the three functions above in sequence and returns the summary dictionary.
Call data_pipeline(arr) and print each key and its value from the result.

This is the last answer to put in warmups_01.py. Congrats!!!

The next question will be in prefect_warmup.py, but will implement the same functionality using Prefect instead of plain Python.
"""

