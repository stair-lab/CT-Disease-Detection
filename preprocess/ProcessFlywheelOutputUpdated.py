import pandas as pd
import json
import os
import numpy as np
from ast import literal_eval

# Define expected biomarker ranges
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

# Define the expected schema for session.info based on the updated CSV
expected_keys = {
    "StudyInfo.Accession",
    "StudyInfo.MRN",
    "StudyInfo.SeriesInfo",
    "LiverValues.LiverMedianHU",
    "LiverValues.LiverVolume",
    "SpleenValues.SpleenMedianHU",
    "SpleenValues.SpleenVolume",
    "KidneyValues.KidneyMedianHU",
    "KidneyValues.KidneyVolume",
    "MuscleValues.L1MuscleMeanHU",
    "MuscleValues.L1MuscleArea",
    "MuscleValues.L3MuscleMeanHU",
    "MuscleValues.L3MuscleArea",
    "CalciumScoring.AbdominalAgatston",
    "CalciumScoring.TotalAorticAgatston",
    "BMDL1Values.BMDL1StandardHU",
    "BMDL1Values.BMDL1HighSensitivityHU",
    "BMDL3Values.BMDL3StandardHU",
    "BMDL3Values.BMDL3HighSensitivityHU",
    "BMDT10Values.BMDT10StandardHU",
    "BMDT10Values.BMDT10HighSensitivityHU",
    "BMDT12Values.BMDT12StandardHU",
    "BMDT12Values.BMDT12HighSensitivityHU",
    "L1FatValues.L1TATArea",
    "L1FatValues.L1VATArea",
    "L1FatValues.L1SATArea",
    "L1FatValues.L1VATSATRatio",
    "L1FatValues.L1VATMedian",
    "L3FatValues.L3TATArea",
    "L3FatValues.L3VATArea",
    "L3FatValues.L3SATArea",
    "L3FatValues.L3VATSATRatio",
    "L3FatValues.L3VATMedian",
    "T10FatValues.T10TATArea",
    "T10FatValues.T10VATArea",
    "T10FatValues.T10SATArea",
    "T10FatValues.T10VATSATRatio",
    "T10FatValues.T10VATMedian",
    "T12FatValues.T12TATArea",
    "T12FatValues.T12VATArea",
    "T12FatValues.T12SATArea",
    "T12FatValues.T12VATSATRatio",
    "T12FatValues.T12VATMedian",
    "Levels.AorticBifurcation",
    "Levels.AorticHiatus",
    "Levels.L1Level",
    "Levels.L2Level",
    "Levels.L3Level",
    "Levels.L4Level",
    "Levels.L5Level",
    "Levels.T10Level",
    "Levels.T11Level",
    "Levels.T12Level",
    "DICOMHeader.BodyPartExamined",
    "DICOMHeader.ContrastBolusAgent",
    "DICOMHeader.ContrastBolusRoute",
    "DICOMHeader.ConvolutionKernel",
    "DICOMHeader.Exposure",
    "DICOMHeader.FilterType",
    "DICOMHeader.ImageType",
    "DICOMHeader.InputSliceThickness",
    "DICOMHeader.InputSpacingBetweenSlices",
    "DICOMHeader.KVP",
    "DICOMHeader.Laterality",
    "DICOMHeader.Manufacturer",
    "DICOMHeader.ManufacturerModelName",
    "DICOMHeader.PatientWeight",
    "DICOMHeader.ProtocolName",
    "DICOMHeader.ReconstructionDiameter",
    "DICOMHeader.RevolutionTime",
    "DICOMHeader.ScanOptions",
    "DICOMHeader.SoftwareVersions",
    "DICOMHeader.SpiralPitchFactor",
    "DICOMHeader.XRayTubeCurrent"
}


# Function to safely evaluate JSON-like strings
def safe_literal_eval(s):
    try:
        return literal_eval(s)
    except (ValueError, SyntaxError):
        return {}  # Return an empty dict if parsing fails

# Flatten JSON and drop rows that don't match the expected schema
def flatten_json_and_validate(row):
    try:
        row_data = pd.json_normalize(row, errors='ignore')
        row_keys = set(row_data.columns)
        
        # Check if row keys match the expected schema
        if not expected_keys.issubset(row_keys):
            return None  # Drop rows that don't match the expected schema
        return row_data
    except Exception:
        return None

# Function to rename columns after flattening
def rename_columns(df):
    new_columns = {}
    for col in df.columns:
        parts = col.split('.')
        if len(parts) == 3:
            new_columns[col] = f"{parts[1]}_{parts[2]}"
    df.rename(columns=new_columns, inplace=True)

# Correct out-of-range values
def correct_out_of_range_values(df):
    for column_name, (lower_limit, upper_limit) in biomarker_ranges.items():
        if column_name in df.columns:
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
            df[column_name] = df[column_name].apply(
                lambda x: "OutOfRange_Missing" if pd.isna(x) or x < lower_limit or x > upper_limit else x
            )
    return df

# Process the DataFrame
def process_dataframe(df):
    zero_check_columns = [
        'LiverValues_LiverMedianHU', 'LiverValues_LiverVolume',
        'SpleenValues_SpleenMedianHU', 'SpleenValues_SpleenVolume',
        'KidneyValues_KidneyMedianHU', 'KidneyValues_KidneyVolume',
        'MuscleValues_L1MuscleMeanHU', 'MuscleValues_L1MuscleArea'
    ]
    
    # Create IVContrastPresent column based on conditions
    conditions = [
        (df['SpleenValues_SpleenMedianHU'] > 65) & (df['SpleenValues_SpleenVolume'] > 50) & (df['SpleenValues_SpleenVolume'] < 6000),
        (df['SpleenValues_SpleenVolume'] > 50) & (df['SpleenValues_SpleenVolume'] < 6000),
        (df['KidneyValues_KidneyMedianHU'] > 45) & (df['KidneyValues_KidneyVolume'] > -10) & (df['KidneyValues_KidneyVolume'] < 750),
        (df['KidneyValues_KidneyVolume'] > -10) & (df['KidneyValues_KidneyVolume'] < 750)
    ]
    choices = ['Yes', 'No', 'Yes', 'No']
    df['IVContrastPresent'] = np.select(conditions, choices, default='Unsure')

    # Drop rows where all zero_check_columns have zero values
    df_non_zero = df[~(df[zero_check_columns] == 0).all(axis=1)].copy()

    # Replace zeros with "Missing" when more than 10 columns have zeros
    mask = (df_non_zero[zero_check_columns] == 0).sum(axis=1) > 10
    df_non_zero.loc[mask, zero_check_columns] = df_non_zero.loc[mask, zero_check_columns].replace(0, 'Missing')

    # Filter out rows where 'LiverValues_LiverMedianHU' is missing
    filtered_df = df_non_zero[df_non_zero['LiverValues_LiverMedianHU'] != 'Missing'].copy()

    return filtered_df

# Sort columns to maintain order
def sort_columns(df):
    studyinfo_columns = [col for col in df.columns if 'studyinfo' in col.lower()]
    level_dicom_columns = [col for col in df.columns if 'level' in col.lower() or 'dicom' in col.lower()]
    remaining_columns = [col for col in df.columns if col not in studyinfo_columns and col not in level_dicom_columns]
    new_column_order = studyinfo_columns + remaining_columns + level_dicom_columns
    sorted_df = df[new_column_order]
    return sorted_df

# Set working directory
new_directory = '/lfs/turing1/0/mahmedc/CT-Disease-Detection'
os.chdir(new_directory)

# Load CSV into a DataFrame
csv_file_path = 'Biomarker_View.csv'
df = pd.read_csv(csv_file_path)

# Convert session.info to dictionary
df['session.info'] = df['session.info'].apply(safe_literal_eval)

# Flatten and validate session.info
flattened_data_list = []
for row in df['session.info']:
    flattened_row = flatten_json_and_validate(row)
    if flattened_row is not None:
        flattened_data_list.append(flattened_row)

# Concatenate flattened data and drop invalid rows
if flattened_data_list:
    flattened_data = pd.concat(flattened_data_list, ignore_index=True, sort=False)
    df = df.drop('session.info', axis=1)
    result_df = pd.concat([df, flattened_data], axis=1)
else:
    print("No valid rows matching the expected schema.")
    result_df = pd.DataFrame()

# Rename columns
rename_columns(result_df)

# Correct out-of-range values
result_df = correct_out_of_range_values(result_df)

# Process DataFrame to clean up invalid or zero-filled rows
result_df = process_dataframe(result_df)

# Sort columns in preferred order
result_df = sort_columns(result_df)

# Save the cleaned and flattened DataFrame to a new CSV
output_csv_file_path = 'BiomarkerView_Updated.csv'
result_df.to_csv(output_csv_file_path, index=False)
