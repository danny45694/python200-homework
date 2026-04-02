import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import pearsonr
from pandas.api.types import is_numeric_dtype
from prefect import task, flow
from prefect.logging import get_run_logger 
import seaborn as sns

#---------------------------------Task 1------------------------------

#Function to add files to file_list

@task
def file_path():
    logger = get_run_logger()
    file_list = []
    folder = "happiness_project"
    if not os.path.exists(folder):
        logger.warning("Folder not found: %s", folder)
        return []
    for file in os.listdir(folder):
        full_path = os.path.join(folder, file)
        file_list.append(full_path)
    logger.info("Found %d files in %s", len(file_list), folder)
    return file_list
    
#Convert each csv file into a dataframe, add column with index=year. Add to a list

@task
def convert_list(file_list):
    logger = get_run_logger()
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
    logger.info("Files successfully converted to dataframe and added to list")  
    return converted_list


#Merge dataframes together
@task
def merge_dataframes(converted_list):
    logger = get_run_logger()
    merged_dataframe = pd.concat(converted_list)
    logger.info("Dataframes in list successfully merged together")
    return merged_dataframe

@task
def output_filepath():
    logger = get_run_logger()
    logger.info("Calling or creating output folder")
    # Specify file path using the os module
    output_dir = "outputs"

    # can swap for user prompt to adjust name
    string = "merged_happiness.csv"
    #Check if it exists
    os.makedirs(output_dir, exist_ok=True)
    #Create complete filepath + filename
    output_filepath = os.path.join(output_dir, string)
    logger.info("output folder successfully verified. File path successfully completed")
    return output_filepath

@task
def write_csv(merged_dataframe, output_filepath):
    logger = get_run_logger()
    #Writing CSV file
    merged_dataframe.to_csv(output_filepath, index=False)

    logger.info(f"CSV file successfully written to {output_filepath} using pandas")
    

@task(retries=3, retry_delay_seconds=2)
def create_update_csv(merged, filepath):
    logger = get_run_logger()
    write_csv(merged, filepath)
    logger.info("Created CSV file from merged dataframe. Returning dataframe for use")


#----------------------------- Task 2 -----------------------------

@task
def happiness_stats(dataframe):
    logger = get_run_logger()
    mean = dataframe['Happiness score'].mean()
    median = dataframe['Happiness score'].median()
    std = dataframe['Happiness score'].std()
    logger.info("Mean: %.2f | Median: %.2f | Std: %.2f", mean, median, std)
    return mean, median, std


#---------------------------- Task 3 -----------------------------

#Specific output function for saving plots
@task
def output_file(string):
    output_dir = "outputs"
    output_filepath = os.path.join(output_dir, string)
    plt.savefig(output_filepath)


# I need to get the mean happiness score of all countries in a file

@task
def hist_distributions(dataframe):
    mean_by_year = dataframe.groupby(dataframe['Year'])['Happiness score'].mean()
    #print(mean_by_year)
    plt.figure()

    plt.hist(mean_by_year, bins=5)
    plt.title('How happiness scores are distributed worldwide over the past nine years')
    plt.xlabel("Happiness worldwide")
    plt.ylabel("Happiness score")

    #Save to outputs folder
    output_file('happiness_histogram.png')

@task
def happiness_across_years(dataframe):

    happiness_score = []
    labels = []
    for year, group in dataframe.groupby('Year', sort=False):
        happiness_score.append(group["Happiness score"].values)
        labels.append(year)

    plt.boxplot(happiness_score, labels=labels)
    plt.show()
    output_file('happiness_by_year.png')

@task
def gdp_vs_happiness(dataframe):

    dataframe.plot.scatter(x="GDP per capita", y="Happiness score")
    plt.title("GDP and Happiness")
    plt.xlabel("GDP")
    plt.ylabel("Happiness")
    output_file("gdp_vs_happiness.png")

    #Display the plot
    plt.show()


@task
def correlation_heatmap(dataframe):
    data = dataframe[["Happiness score", "GDP per capita", "Social support", "Freedom to make life choices", "Generosity", "Perceptions of corruption"]].copy()

    corr_matrix = data.corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True)
    output_file("correlation_heatmap.png")
    plt.show()

#Task 4: Hypothesis Testing

#Compare happiness scores from 2019 to 2020

@task
def testing_happiness(dataframe):
    groups = {}

    # Group data
    for year, group in dataframe.groupby("Year"):
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

@task
def region_compare(dataframe):
    groups ={}

    #Group Data
    for Country, group in dataframe.groupby("Country"):
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

@task
def comparison_to_happiness(dataframe):
    x = dataframe["Happiness score"]
    results_list = []
    number_of_tests = 0
    for col in dataframe.columns:
        if is_numeric_dtype(dataframe[col]):
            if col == "Happiness score":
                continue
            y = dataframe[col]
            r, p = pearsonr(x, y)
            print(f"Correlation: {r}, p-value: {p}")
            number_of_tests += 1

            correlation_to_happiness = {
            "column": col,
            "correlation": r,
            "p-value": p
            }

            results_list.append(correlation_to_happiness)

            print("Correlation:", round(r, 2))
            print("p-value:,", round(p, 4))
    return results_list, number_of_tests
    
@task
def adjusted_alpha_test(dataframe):
    results_list, number_of_tests = comparison_to_happiness(dataframe)
    adjusted_alpha = 0.05 / number_of_tests

    for item in results_list:
        item['adjusted_alpha'] = item['p-value'] < adjusted_alpha
    return results_list


@flow
def happiness_pipeline():
    logger = get_run_logger()

    # --- Task 1: Build the dataset ---
    logger.info("Starting pipeline: loading files")
    file_list = file_path()
    converted = convert_list(file_list)
    dataframe = merge_dataframes(converted)
    filepath = output_filepath()
    write_csv(dataframe, filepath)

    # --- Task 2: Descriptive stats ---
    logger.info("Descriptive statistics")
    happiness_stats(dataframe)

    # --- Task 3: Visualisations ---
    logger.info("Generating plots")
    hist_distributions(dataframe)
    happiness_across_years(dataframe)
    gdp_vs_happiness(dataframe)
    correlation_heatmap(dataframe)

    # --- Task 4: Hypothesis testing ---
    logger.info("Running hypothesis tests")
    testing_happiness(dataframe)
    region_compare(dataframe)

    # --- Task 5: Correlations ---
    logger.info("Running correlation analysis")
    adjusted_alpha_test(dataframe)

if __name__ == "__main__":
    happiness_pipeline()
  
