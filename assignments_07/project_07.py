from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import re
from prefect import task, flow
from pandas.api.types import is_numeric_dtype
from prefect.logging import get_run_logger
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from openai import OpenAI

# smolagents imports
from smolagents import ToolCallingAgent, OpenAIServerModel, tool
from smolagents import CodeAgent

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()


# ------------------------------------- Pre-task --------------------------------------

def convert_list(folder):
    #logger = get_run_logger()
    converted_list = []
    for file in folder:
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

def merge_dataframes(converted_list):
    logger = get_run_logger()
    merged_dataframe = pd.concat(converted_list)
    logger.info("Dataframes in list successfully merged together")
    return merged_dataframe


DATA_PATH = Path.cwd().parent / "assignments_01" / "outputs"

#Check for outputs existence
if DATA_PATH.exists():
    print("Outputs folder found!")
    for file in DATA_PATH.glob("*"):
        print(file.name)
else:
    # 3. Fallback path if outputs is missing
    folder = Path.cwd().parent / "assignments_01" / "happiness_project"
    
    if folder.exists():
        print("Outputs not found. Happiness_project folder found.")
        converted_list = convert_list(folder)
        merged_dataframe = merge_dataframes(converted_list)
        return merged_dataframe
    else:
        print("Neither folder was found.")

# Add the code to merge the csvs together and store in Dataframe




#Global Dataframe
df = None


@tool
def load_happiness_data(DATA_PATH) -> dict:
    """
    Load the csv file from DATA_PATH, store it in the global df. 

    filename can be "merged_happiness" or "merged_happiness.csv"

    """
    global df
    df = pd.read_csv(DATA_PATH)
    return {"shape": df.shape, "columns": df.columns.tolist()}
 

 # Need to move this elsewhere
"""
    Args:
        filename: CSV filename in assignments_01/outputs/. You can pass "merged_happiness" or merged_happiness.csv". 
        If merged_happiness does not exist.
        filename: CSV files in assignments_01/happiness_project. Iterate through each file, merging all the yearly CSV files.

    Returns:
        Store the result in the global "df" variable. Return a dict with "shape" and "columns"
    """
     

def get_columns(self):
    """
    Return column names for the currently loaded CSV.
    """
    error = self._ensure_loaded()
    if error:
        return error
    return self.df.columns.tolist()


@tool
def summarize_column(column: str) -> dict:

    """
    Return basic summary stats for one or more columns.

    If columns is None, summarize all columns.
    Uses pandas.describe(include="all") to stay simple and readable.

    """

    if df is None:
        return {"error": "No data loaded yet. Please run load_happiness_data first."}
    
    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}
    
    return df[column].describe().to_dict()


@tool
def summarize_column(column: str) -> dict:
    """
    Return descriptive statistics for a single column in the loaded dataset

    Args:
        columns: Column names to summarize. If columns is None, summarize all columns.

    Returns:
        A dict of summary statistics (from pandas.describe), or an error dict.
    """

    if column is None:
        data = df
    else:
        missing = str not in df.columns
        if missing:
            return {"error": f"This column is not in the data: {missing}"}
    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame using scipy.stats.pearsonr.
        Return the col1, col2, correlation coefficient as pearson_r, and p-value in a dict.

        Args:
            col1: Column 1 used for the Pearson correlation
            col2: Column 2 used for the Pearson correlation

            Returns:
                 A dict with col1, col2, pearson_r, and p_value as keys and their respective values.

        """

        for col in [col1, col2]:
            if col not in df.columns:
                return {"error": f"'{col}' is not a column. Options: {df.columns.tolist()}"}
            
        data1 = df[col1]
        data2 = df[col2]

        corr, p = pearsonr(data1, data2)
        pearson_r = round(corr, 4)
        p_value = round(p, 4)


        result = {
            "col1": data1,
            "col2": data2,
            "pearson_r": pearson_r,
            "p_value": p_value
        }
        return result


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    ...
    """

    if not isinstance(column, str) or not isinstance(year, int):
        return {"error": "bad input. Enter in a column name as string and year as integer"}
    
    df_filtered = df[(df['year']== year)]
    sort = df.sort_values(by=[column], ascending=False)
    top_n_rows = df.iloc[0: n]
    
    return {"country": column}
    


# ------------------------------------- Task 2 -------------------------------------------

from smolagents import CodeAgent, OpenAIServerModel

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)





# ------------------------------------- Task 3 ------------------------------------------


queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the happiness_score column.",
    "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
]

for query in queries:
    print(f"\n--- Query: {query} ---")
    response = agent.run(query, reset=False)
    print(response)



# ------------------------------------- Task 4 --------------------------------------

"""
# My query 1
my_query_1 = "..."   # replace with your question
response_1 = agent.run(my_query_1, reset=False)
print(response_1)
# Comment: Did this trigger tool use, code generation, or both?

# My query 2
my_query_2 = "..."   # replace with your question
response_2 = agent.run(my_query_2, reset=False)
print(response_2)
# Comment: Did this trigger tool use, code generation, or both?

"""