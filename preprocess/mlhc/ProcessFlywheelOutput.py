import pandas as pd
# from pandas.io.json import json_normalize  # Old way, not needed now
import json
import os
import numpy as np
from ast import literal_eval

biomarker_ranges = {
    "LiverValues_LiverMedianHU": (-50, 300),
    "LiverValues_LiverVolume": (100, 5000),
    "SpleenValues_SpleenMedianHU": (10, 500),
    "SpleenValues_SpleenVolume": (50, 6000),
    "KidneyValues_KidneyMedianHU": (5, 300),
    "KidneyValues_KidneyVolume": (50, 750),
    "MuscleValues_L1MuscleMeanHU": (-50, 200),
    "MuscleValues_L1MuscleArea": (25, 500),
    "MuscleValues_L3MuscleMeanHU": (-50, 200),
    "MuscleValues_L3MuscleArea": (25, 500),
    "CalciumScoring_AbdominalAgatston": (0, 40000),
    "BMDL1Values_BMDL1StandardHU": (-50, 1200),
    "BMDL1Values_BMDL1HighSensitivityHU": (-50, 1200),
    "BMDL3Values_BMDL3StandardHU": (-50, 1200),
    "BMDL3Values_BMDL3HighSensitivityHU": (-50, 1200),
    "L1FatValues_L1TATArea": (0.1, 1500),
    "L1FatValues_L1VATArea": (0, 1200),
    "L1FatValues_L1SATArea": (0.1, 1000),
    "L1FatValues_L1VATSATRatio": (0, 5),
    "L1FatValues_L1VATMedian": (-120, -30),
    "L3FatValues_L3VATMedian": (-120, -30),
    "L3FatValues_L3TATArea": (0.1, 1500),
    "L3FatValues_L3VATArea": (0, 1200),
    "L3FatValues_L3SATArea": (0.1, 1000),
    "L3FatValues_L3VATSATRatio": (0, 5),
    "T10FatValues_T10VATMedian": (-120, -30),
    "T10FatValues_T10TATArea": (0.1, 1500),
    "T10FatValues_T10VATArea": (0, 1200),
    "T10FatValues_T10SATArea": (0.1, 1000),
    "T10FatValues_T10VATSATRatio": (0, 5),
    "T12FatValues_T12VATMedian": (-120, -30),
    "T12FatValues_T12TATArea": (0.1, 1500),
    "T12FatValues_T12VATArea": (0, 1200),
    "T12FatValues_T12SATArea": (0.1, 1000),
    "T12FatValues_T12VATSATRatio": (0, 5)
}

# Sample function to safely evaluate the JSON-like strings
def safe_literal_eval(s):
    try:
        return literal_eval(s)
    except ValueError:
        return {}  # Return an empty dict in case of evaluation error

# Function to flatten JSON contained in each row of the specified column
# def flatten_json(row):
#     return pd.json_normalize(row)

# Apply the function to flatten the JSON and handle varying schemas
def flatten_json(row):
    try:
        # Use json_normalize to flatten and ensure errors are ignored for missing keys
        return pd.json_normalize(row, errors='ignore')
    except Exception as e:
        return pd.DataFrame()  # Return an empty DataFrame on failure
    
def rename_columns(df):
    new_columns = {}
    for col in df.columns:
        # Split the column name by dot and check if it splits into three parts
        parts = col.split('.')
        if len(parts) == 3:
            # Construct the new column name format as B_C
            new_columns[col] = f"{parts[1]}_{parts[2]}"
    # Rename the columns using the dictionary
    df.rename(columns=new_columns, inplace=True) 

def correct_out_of_range_values(df, correction_df):
    """
    Corrects values in df based on the lower and upper limits specified in correction_df.
    
    Parameters:
    - df: The main DataFrame to be corrected.
    - correction_df: A DataFrame with columns ['ColumnName', 'LowerLimit', 'UpperLimit']
    
    Returns:
    - df: The corrected DataFrame with out of range values replaced by "OutOfRange".
    """
    for column_name, (lower_limit, upper_limit) in biomarker_ranges.items():
       
        # Check if column exists in df
        if column_name in df.columns:
            # Convert column values to numeric, coercing errors to NaN
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
            # Replace values outside the range or NaN with "OutOfRange"
            df[column_name] = df[column_name].apply(lambda x: "OutOfRange_Missing" if pd.isna(x) or x < lower_limit or x > upper_limit else x)
        else:
            print(f"Column {column_name} does not exist in the DataFrame.")
            
    return df

def process_dataframe(df):
    # Define columns for zero-check
    zero_check_columns = [
        'LiverValues_LiverMedianHU', 'LiverValues_LiverVolume', 
        'SpleenValues_SpleenMedianHU', 'SpleenValues_SpleenVolume', 
        'KidneyValues_KidneyMedianHU', 'KidneyValues_KidneyVolume', 
        'MuscleValues_L1MuscleMeanHU', 'MuscleValues_L1MuscleArea', 
        'MuscleValues_L3MuscleMeanHU', 'MuscleValues_L3MuscleArea', 
        'CalciumScoring_AbdominalAgatston', 'BMDL1Values_BMDL1StandardHU',
        'BMDL1Values_BMDL1HighSensitivityHU','BMDL3Values_BMDL3StandardHU',
        'BMDL3Values_BMDL3HighSensitivityHU', 'L1FatValues_L1TATArea', 
        'L1FatValues_L1VATArea', 'L1FatValues_L1SATArea', 'L1FatValues_L1VATSATRatio', 
        'L1FatValues_L1VATMedian', 'L3FatValues_L3VATMedian', 'L3FatValues_L3TATArea', 
        'L3FatValues_L3VATArea', 'L3FatValues_L3SATArea', 'L3FatValues_L3VATSATRatio'
    ]
    
    df['SpleenValues_SpleenMedianHU'] = pd.to_numeric(df['SpleenValues_SpleenMedianHU'], errors='coerce')
    df['SpleenValues_SpleenVolume'] = pd.to_numeric(df['SpleenValues_SpleenVolume'], errors='coerce')
    df['KidneyValues_KidneyMedianHU'] = pd.to_numeric(df['KidneyValues_KidneyMedianHU'], errors='coerce')
    df['KidneyValues_KidneyVolume'] = pd.to_numeric(df['KidneyValues_KidneyVolume'], errors='coerce')

    # Filter out rows with all zeros in specified columns
    #df_non_zero = df[~(df[zero_check_columns] == 0).all(axis=1)].copy()

    # Assuming df is your original DataFrame
    # Assuming zero_check_columns is a list of column names to check for zeros

    # Step 1: Create the hascontrast column in the original dataframe
    conditions = [
        (df['SpleenValues_SpleenMedianHU'] > 65) & (df['SpleenValues_SpleenVolume'] > 50) & (df['SpleenValues_SpleenVolume'] < 6000),
        (df['SpleenValues_SpleenVolume'] > 50) & (df['SpleenValues_SpleenVolume'] < 6000),
        (df['KidneyValues_KidneyMedianHU'] > 45) & (df['KidneyValues_KidneyVolume'] > -10) & (df['KidneyValues_KidneyVolume'] < 750),
        (df['KidneyValues_KidneyVolume'] > -10) & (df['KidneyValues_KidneyVolume'] < 750)
    ]

    choices = ['Yes', 'No', 'Yes','No']
    df['IVContrastPresent'] = np.select(conditions, choices, default='Unsure')

    # Step 2: Create df_non_zero by dropping rows where all specified columns have zero values
    df_non_zero = df[~(df[zero_check_columns] == 0).all(axis=1)].copy()

    # Step 3: In df_non_zero, replace zeros with "Missing" where more than 10 specified columns have zeros
    mask = (df_non_zero[zero_check_columns] == 0).sum(axis=1) > 10
    df_non_zero.loc[mask, zero_check_columns] = df_non_zero.loc[mask, zero_check_columns].replace(0, 'Missing')

    # Filter out rows with "Missing" in 'LiverValues_LiverMedianHU'
    filtered_df = df_non_zero[df_non_zero['LiverValues_LiverMedianHU'] != 'Missing'].copy()
 
    return filtered_df

def sort_columns(df):
    # Convert all column names to lowercase for case-insensitive comparison
    lower_columns = df.columns.str.lower()
    
    # Identify columns that contain 'studyinfo', 'level', or 'dicom'
    studyinfo_columns = [col for col in df.columns if 'studyinfo' in col.lower()]
    level_dicom_columns = [col for col in df.columns if 'level' in col.lower() or 'dicom' in col.lower()]
    
    # Exclude the identified columns to get the remaining columns
    remaining_columns = [col for col in df.columns if col not in studyinfo_columns and col not in level_dicom_columns]
    
    # Concatenate the 'studyinfo' columns, remaining columns, and then 'level'/'dicom' columns for the new order
    new_column_order = studyinfo_columns + remaining_columns + level_dicom_columns
    
    # Reorder the DataFrame's columns
    sorted_df = df[new_column_order]
    
    return sorted_df
# Step 0: Change the working directory
new_directory = '/lfs/turing1/0/mahmedc/CT-Disease-Detection'  # Update this to the directory containing your CSV file
os.chdir(new_directory)

# Step 1: Load the CSV into a DNew_biomarker_folderataFrame
csv_file_path = 'Biomarker_View.csv'  # Now you can just use the file name
df = pd.read_csv(csv_file_path)
df['session.info'] = df['session.info'].apply(safe_literal_eval)

# Apply the function to each row in the column containing JSON strings
# This creates a Series of DataFrames, so we use pd.concat to merge these into a single DataFrame
# flattened_data = pd.concat(df['session.info'].apply(flatten_json).tolist(), ignore_index=True)

# Now, you have the option to concatenate this flattened data back with the original DataFrame
# Depending on your needs, you might want to drop the original 'json_column' or keep it
# df = df.drop('session.info', axis=1)
# result_df = pd.concat([df, flattened_data], axis=1)

# Flatten JSON data and handle schema variations
flattened_data_list = []
for row in df['session.info']:
    flattened_row = flatten_json(row)
    flattened_data_list.append(flattened_row)

# Concatenate all the flattened rows while handling missing keys dynamically
flattened_data = pd.concat(flattened_data_list, ignore_index=True, sort=False)

# Drop 'session.info' and merge flattened columns with the original DataFrame
df = df.drop('session.info', axis=1)
result_df = pd.concat([df, flattened_data], axis=1)


# Function to rename columns


# Apply the function to your DataFrame
rename_columns(result_df)

result_df = correct_out_of_range_values(result_df, biomarker_ranges)
result_df = process_dataframe(result_df)
# columns_to_drop = ['errors','BMDT10Values_BMDT10HighSensitivityHU',	'BMDT10Values_BMDT10StandardHU',	'BMDT12Values_BMDT12HighSensitivityHU',	'BMDT12Values_BMDT12StandardHU',	'CalciumScoring_TotalAorticAgatston',	'CalculiValues_CalculiMedianHU',	'CalculiValues_CalculiVolume',	'T10FatValues_T10SATArea', 'T10FatValues_T10TATArea',	'T10FatValues_T10VATArea',	'T10FatValues_T10VATMedian',	'T10FatValues_T10VATSATRatio',	'T12FatValues_T12SATArea',	'T12FatValues_T12TATArea',	'T12FatValues_T12VATArea',	'T12FatValues_T12VATMedian',	'T12FatValues_T12VATSATRatio', 'BMDL3Values_BMDL3HighSensitivityHU','BMDL3Values_BMDL3StandardHU','L1FatValues_L1SATArea','L1FatValues_L1TATArea','L1FatValues_L1VATArea','L1FatValues_L1VATMedian','L1FatValues_L1VATSATRatio','MuscleValues_L1MuscleArea','MuscleValues_L1MuscleMeanHU',]
# result_df.drop(columns=columns_to_drop, inplace=True)
result_df=sort_columns(result_df)
# Step 4: Save the flattened DataFrame to a new CSV file
output_csv_file_path = 'Duly_2025_BiomarkerView_Updated.csv'  # You can also just use the file name here
result_df.to_csv(output_csv_file_path, index=False)
