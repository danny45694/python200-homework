from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import re
from pandas.api.types import is_numeric_dtype
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from dotenv import load_dotenv
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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "assignments_01", "outputs", "merged_happiness.csv")
print(BASE_DIR)
print(DATA_PATH)

from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import re
from pandas.api.types import is_numeric_dtype
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

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(base, "assignments_01", "outputs", "merged_happiness.csv")
FALLBACK_FOLDER = os.path.join(base, "assignments_01", "happiness")

# Global DataFrame placeholder — must be at top level so tools can access it
df = None


# ------------------------------------- Pre-task --------------------------------------

def file_path(FALLBACK_FOLDER):
    file_list = []
    for file in os.listdir(FALLBACK_FOLDER):
        full_path = os.path.join(FALLBACK_FOLDER, file)
        file_list.append(full_path)
    return file_list


def convert_list(file_list):
    converted_list = []
    for file in file_list:
        # Find year in each file
        year = re.findall(r'\d+', file)

        # Pandas read csv file
        df = pd.read_csv(file, sep=";", decimal=",")
        # Add year to csv files
        df['Year'] = int(year[0])

        # 2024 has "Ladder score" not "Happiness score", need to modify dataframe
        if int(year[0]) == 2024:
            df.rename(columns={"Ladder score": "Happiness score"}, inplace=True)

        # New list with created data frames
        converted_list.append(df)
    return converted_list


def merge_dataframes(converted_list):
    merged_dataframe = pd.concat(converted_list, ignore_index=True)
    return merged_dataframe


@tool
def load_happiness_data() -> dict:
    """
    Load the CSV file from DATA_PATH into the global df.
    First checks if DATA_PATH is a valid path. If not, falls back to FALLBACK_FOLDER
    which contains raw data files and recreates the dataframe.
    Returns a dict with 'shape' and 'columns' of the loaded data.
    """
    global df

    if not os.path.exists(DATA_PATH):
        print("Outputs not found. Attempting fallback to happiness_project folder...")
        if not os.path.exists(FALLBACK_FOLDER):
            return {"error": "Neither the merged file nor raw data folder was found."}
        # FIX: assign directly to df instead of to DATA_PATH
        file_list = file_path(FALLBACK_FOLDER)
        converted_list = convert_list(file_list)
        df = merge_dataframes(converted_list)
    else:
        df = pd.read_csv(DATA_PATH)

    if len(df) == 0:
        return {"error": "Loaded data is empty. Double check path is correct."}

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
    Compute the Pearson correlation between two columns in the loaded DataFrame.
    Returns col1, col2, pearson_r, and p_value in a dict.
    Args:
        col1: Column 1 used for the Pearson correlation.
        col2: Column 2 used for the Pearson correlation.
    """
    if df is None:
        return {"error": "No data loaded yet. Please run load_happiness_data first."}

    for col in [col1, col2]:
        if col not in df.columns:
            return {"error": f"'{col}' is not a column. Options: {df.columns.tolist()}"}

    # Clean data
    clean_df = df[[col1, col2]].dropna()
    corr, p = pearsonr(clean_df[col1], clean_df[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(corr, 4),
        "p_value": round(p, 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """
    Return the top N countries ranked by a given column for a specific year.
    Args:
        column: Column name to sort values by.
        year: Target year as an integer.
        n: Number of top results to return.
    """
    if df is None:
        return {"error": "No data loaded yet. Please run load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"Column '{column}' missing. Options: {df.columns.tolist()}"}

    df_filtered = df[df['Year'] == year].copy()
    if len(df_filtered) == 0:
        return {"error": f"No data found for year {year}."}

    result = (
        df_filtered
        .sort_values(by=column, ascending=False)
        .iloc[:n][['Country', column]]
    )
    return result.to_dict(orient='records')


# ------------------------------------- Task 2 -------------------------------------------

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries.
IMPORTANT RULES:
- Tools return all the data you need. Use their return values directly.
- NEVER access or reference a variable called df. It does not exist in your environment.
- Do NOT reload data yourself with pandas after calling load_happiness_data.
- Write Python code only for plotting or computations not covered by tools.
- The load_happiness_data tool returns shape and columns directly. Use that return value.
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

if __name__ == "__main__":
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
    # my_query_1 = "..."   # replace with your question
    # response_1 = agent.run(my_query_1, reset=False)
    # print(response_1)
    # Comment: Did this trigger tool use, code generation, or both?

    # My query 2
    # my_query_2 = "..."   # replace with your question
    # response_2 = agent.run(my_query_2, reset=False)
    # print(response_2)
    # Comment: Did this trigger tool use, code generation, or both?


# -------------------------------------- Task 5 -------------------------------------

# --- Reflection ---

"""
 1. 

"""