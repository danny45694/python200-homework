from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import re

from scipy.stats import pearsonr
import os
from pathlib import Path
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

#Global Dataframe placeholder
df = None

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(base, "assignments_01", "outputs", "merged_happiness.csv")

FALLBACK_FOLDER = os.path.join(base, "assignments_01", "happiness_project")

# ------------------------------------- Pre-task --------------------------------------

#Created merged_happiness.csv if it doesn't exist. 
def file_path(FALLBACK_FOLDER):
    file_list = []
    for file in os.listdir(FALLBACK_FOLDER):
        full_path = os.path.join(FALLBACK_FOLDER, file)
        file_list.append(full_path)
    return file_list

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
    return converted_list

def merge_dataframes(converted_list):
    merged_dataframe = pd.concat(converted_list, ignore_index=True)
    return merged_dataframe


@tool
def load_happiness_data() -> dict:
    """
    Load the csv file from DATA_PATH, generates a dataframe and stores it in global "df" variable.  

    Args:
        path_str: String path to the CSV dataset file.

    """
    global df
    global base
    global DATA_PATH
    global FALLBACK_FOLDER

    if not os.path.exists(DATA_PATH):
        print("Outputs not found. Attempting fallback to happiness_project folder...")
        if not os.path.exists(FALLBACK_FOLDER):
            return {"error": "Neither the merged file nor raw data folder was found."}
        file_list = file_path(FALLBACK_FOLDER)
        converted_list = convert_list(file_list)
        merged_dataframe = merge_dataframes(converted_list)
        DATA_PATH = merged_dataframe
    

    df = pd.read_csv(DATA_PATH)
    if len(df) == 0:
        return {"error": "DATA_PATH is empty. Double check path is correct"}
    return {"shape": df.shape, "columns": df.columns.tolist()}

@tool
def summarize_column(column: str) -> dict:

    """
    Return basic descriptive stats for a single column.
    Args:
        column: The exact name of the column to summarize. 
    """

    if df is None:
        return {"error": "No data loaded yet. Please run load_happiness_data first."}
    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}
    return df[column].describe().to_dict()



@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame using scipy.stats.pearsonr.
    Load col1, col2, correlation coefficient as pearson_r, and p-value as keys in a dict. Return the dict

    Args:
        col1: Column 1 used for the Pearson correlation
        col2: Column 2 used for the Pearson correlation

    Returns:
            A dict with col1, col2, pearson_r, and p_value as keys and their respective values.
    """

    if df is None:
        return {"error": "No data loaded yet"}

    for col in [col1, col2]:
        if col not in df.columns:
            return {"error": f"'{col}' is not a column. Options: {df.columns.tolist()}"}
        

    corr, p = pearsonr(df[col1], df[col2])

    pearson_r = round(corr, 4)
    p_value = round(p, 4)


    result = {
        "col1": col1,
        "col2": col2,
        "pearson_r": pearson_r,
        "p_value": p_value
    }
    return result


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.
    Args:
        column: Column name to sort values by.
        year: Target year as an integer.
        n: Number of top results to return.
    """

    if df is None:
        return {"error": "No data loaded yet. Please run load_happiness_data first."}
    
    if column not in df.columns:
        return {"error": f"Column {column} missing."}
    
    df_filtered = df[(df['Year'] == year)].copy()
    if len(df_filtered) == 0:
        return {"error": "Year does not exist"}
    result = (df_filtered
              .sort_values(by=column, ascending=False)
              .iloc[:n][['Country', column]])
    

    return result.to_dict(orient='records')

    


# ------------------------------------- Task 2 -------------------------------------------

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


if __name__ == "__main__":



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



    # My query 1
    my_query_1 = "Show me the unhappiest countries in 2020"   # replace with your question
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?

    # My query 2
    my_query_2 = "When year was mean happiness highest? What year was mean happiness lowest?"   
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?




# -------------------------------------- Task 5 -------------------------------------


# --- Reflection ---

"""
 1. Agent says the correlation is statistically significant. P-value has a value of 0 so it appears it was not used correctly. 

"""

