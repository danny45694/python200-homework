import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from pandas.api.types import is_numeric_dtype
#from prefect import task, flow (Comment out for now. Clean print statements)
#from prefect.logging import get_run_logger (use above comment on prefect)
import seaborn as sns

#---------------------------------Task 1------------------------------
folder = "happiness_project"
file_list = []




#Function to add files to file_list

#@task
def file_path(folder):
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        file_list.append(full_path)
    return file_list
    
#print(file_path(folder))

#Convert each csv file into a dataframe, add column with index=year. Add to a list

#@task
def convert_list(file_list):
    converted_list = []
    for file in file_list:
        #Find year in each file
        year = re.findall(r'\d+', file)
        #Pandas read csv file
        df = pd.read_csv(file, sep = ";", decimal=",") 
        #add year to csv files
        df['Year'] = int(year[0])

    #2024 has "Ladder score" not "Happiness score", need to modify dataframe
        if int(year[0]) == 2024:
            df.rename(columns={"Ladder score": "Happiness score"}, inplace=True)
        
        #new list with created data frames
        converted_list.append(df)
    #removed print statements. This should now handle all 10 files    
    return converted_list


#Merge dataframes together
def merge_dataframes(converted_list):
    merged_dataframe = pd.concat(converted_list)
    return merged_dataframe


def output_csv(merged_dataframe):
    # Specify file path using the os module
    output_dir = "outputs"
    #Create the directory if it doesn't exist (Later task will require we do this for the histograms. Need to make this a separate function)
    os.makedirs(output_dir, exist_ok=True)
    output_filepath = os.path.join(output_dir, 'merged_happiness.csv')

    #Writing CSV file
    merged_dataframe.to_csv(output_filepath, index=False)
    print(f"CSV file successfully written to {output_filepath} using pandas")
    

#@task(retries=3, retry_delay_seconds=2)
def create_update_csv(folder, file_list):
    file_list = file_path(folder)
    converted_list = convert_list(file_list)
    merged_dataframe = merge_dataframes(converted_list)
    output_csv(merged_dataframe)
    return merged_dataframe

#create_update_csv(folder, file_list)

#Created a merged_dataframe variable for global use. Easier to test and understand. Also marks the end of Task 1 of mini-project
#merged_dataframe = create_update_csv(folder, file_list)


#-----------------------------Task 2 -----------------------------


"""
Compute and log overall descriptive statistics for happiness_score: mean, median, and standard deviation.

Then compute and log the mean happiness score grouped by year and by region. Looking at the regional breakdown is often the most interesting part of this dataset -- you may already have a hypothesis about which regions rank highest before you run the numbers.

"""
#Creating file_path to merged csv file for later testing

#This may be a bit more redundant than I thought for simple testing. I'll leave it here for now as it may be one of the finishing touches.
 
outputs = 'outputs'
output_file_list = []
def output_path(folder):
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        output_file_list.append(full_path)
    return output_file_list


#Because of the code above, I may need code to re-transform the csv file to a dataframe so I can use it for the next set of tasks

#Simple path to file for next tasks.
merged_file = "outputs\\merged_happiness.csv"

#Creates dataframe from merged file
df = pd.read_csv(merged_file)


#Returning error unhashable type: 'Series'

#Also see this: Warning: Starting with pandas version 4.0 all arguments of mean will be keyword-only.
def happiness_stats(merged_dataframe):
    mean = merged_dataframe['Happiness score'].mean()
    median = merged_dataframe['Happiness score'].median()
    std = merged_dataframe['Happiness score'].std()
    return mean, median, std



#---------------------------- Task 3 -----------------------------

#I want to graph the happiness score of each country by year. How can I accomplish that. 

"""
What we know?


1. The program must know to go by year, not up to 2024
2. There are ~ 157 countries in each csv
3. Says all happiness scores across all years.

Q. A histogram of all happiness scores across all years
Keyword all

Instructions do not specify graph each country. 

Conclusion: Take the mean of all countries, graph that across 9 years

"""

def output_file(string):
    output_dir = "outputs"
    output_filepath = os.path.join(output_dir, string)
    plt.savefig(output_filepath)


# I need to get the mean happiness score of all countries in a file
def hist_distributions(dataframe):
    mean_by_year = dataframe.groupby(dataframe['Year'])['Happiness score'].mean()
    #print(mean_by_year)
    plt.subplot(1,2,1)

    plt.hist(mean_by_year, bins=5)
    plt.title('How happiness scores are distributed worldwide over the past nine years')
    plt.xlabel("Happiness worldwide")
    plt.ylabel("Happiness score")

    #Save to outputs folder

    output_file('happiness_histogram.png')

#Outputs hist_distribution to outputs folder
#hist_distributions(df)

"""
Boxplot comparing happiness score distributions across years (one box per year). Save as happiness_by_year.png
"""

def happiness_across_years(df):

    happiness_score = []
    labels = []

    for year, group in df.groupby('Year', sort=False):
        happiness_score.append(group)
        labels.append(year)

    plt.boxplot(happiness_score, labels=labels)
    plt.show()

    output_file('happiness_by_year.png')

#happiness_across_years(df)


def gdp_vs_happiness(df):

    df.plot.scatter(x="GDP per capita", y="Happiness score")

    plt.title("GDP and Happiness")
    plt.xlabel("GDP")
    plt.ylabel("Happiness")
    
    output_file("gdp_vs_happiness.png")

    #Display the plot
    plt.show()

#gdp_vs_happiness(df)



def correlation_heatmap(dataframe):
    data = dataframe[["Happiness score", "GDP per capita", "Social support", "Freedom to make life choices", "Generosity", "Perceptions of corruption"]].copy()

    corr_matrix = data.corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True)
    output_file("correlation_heatmap.png")
    plt.show()

#correlation_heatmap(df)
#Task 4: Hypothesis Testing


#Compare happiness scores from 2019 to 2020

def testing_happiness(df):
    groups = {}

    # Group data
    for year, group in df.groupby("Year"):
        if year in [2019, 2020]:
            groups[year] = group["Happiness score"]

    # Extract groups
    group_2019 = groups[2019]
    group_2020 = groups[2020]

    # Compute means
    mean_2019 = group_2019.mean()
    mean_2020 = group_2020.mean()

    # T-test 

    t_stat, p_val = stats.ttest_ind(group_2019, group_2020)

    #Log the t-stat, p-value, mean happiness for each group
    print("2019 Mean:", mean_2019)
    print("2020 Mean:", mean_2020)
    print('T-stat:', t_stat)
    print('p-value:', p_val)

    if p_val < 0.05:
        print("The difference is statistically significant.")
    else:
        print("No statistically significant difference detected.")

#Got an error. Need to separate the happiness_scores into 2 groups
#They are all bunched up together. I did not program in an identifier. Similar code to the boxplot problem we did above.

def region_compare():
    groups ={}

    #Group Data
    for Country, group in df.groupby("Country"):
        if Country in ["Switzerland", "Denmark"]:
            groups[Country] = group["Happiness score"]


    group_Switzerland = groups["Switzerland"]
    group_Denmark = groups["Denmark"]

    mean_Switzerland = group_Switzerland.mean()
    mean_Denmark = group_Denmark.mean()

    t_tstat, p_value = stats.ttest_ind(group_Switzerland, group_Denmark)
    print("Switzerland Mean:", mean_Switzerland)
    print("Denmark Mean:", mean_Denmark)
    print("T-stat:", t_tstat)
    print("P-value:", p_value)



#Task 5: Correlation and Multiple Comparisons


x = df["Happiness score"]
number_of_tests = 0


for col in df.columns:
    y = []
    if is_numeric_dtype(df[col]):
        y = df[col].copy
        r, p = pearsonr(x, y)
        print(f"Correlation: {r}, p-value: {p}")
        number_of_tests += 1

        print("Correlation:", round(r, 2))
        print("p-value:,", round(p, 4))

    else:
        continue

adjusted_alpha = 0.05 / number_of_tests
print()
"""

  
#Bonferroni Correction: divide significance threshold by the number of test you ran

#print(adjusted_alpha = 0.05 / number_of_tests)

"""