import pandas as pd

# Load the CSV file
try:
    df = pd.read_csv('dataset/HAM10000_metadata.csv')
    
    print("\n--- SUCCESSS: FILE LOADED ---")
    print("Here are the column names in your file:")
    print(df.columns.tolist())
    
    print("\nHere is the first row of data:")
    print(df.head(1))
    print("-------------------------------\n")
    
except Exception as e:
    print("\nERROR LOADING CSV:")
    print(e)
    