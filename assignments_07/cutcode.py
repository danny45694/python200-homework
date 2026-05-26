from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import re

from scipy.stats import pearsonr
import os







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




def load_happiness_data():
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
    print(f"test: {df}")
    return {"shape": df.shape, "columns": df.columns.tolist()}

load_happiness_data()

print(df)