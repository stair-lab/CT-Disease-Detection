#!/usr/bin/env python3
"""
Modeling Dataset Creation with Train/Val/Test Splits
===================================================

This script processes the biomarkers with HCC codes dataset to create a clean modeling dataset
with appropriate train/validation/test splits for machine learning tasks.

Key Processing Steps:
1. Load and clean the dataset
2. Create standardized column names and formats
3. Process HCC codes as binary variables
4. Create mortality and diagnosis binary variables
5. Process calcium scoring (continuous, binary, and multiclass)
6. Create patient-level splits (70/20/10 train/val/test)
7. Save processed datasets

Author: AI Assistant
Date: 2025-01-27
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

def load_and_examine_data():
    """Load the dataset and examine its structure."""
    print("🚀 MODELING DATASET CREATION WITH SPLITS")
    print("=" * 60)
    
    data_path = "../../datasets/full_data/biomarkers_with_hcc_codes_20250909_062523.csv"
    
    print("📊 Loading dataset...")
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Dataset loaded: {df.shape}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    return df

def process_identifier_and_gender(df):
    """Process unique identifier and gender columns."""
    print("\n🔧 Processing identifier and gender columns...")
    
    # 1. Rename unique identifier column to FILE
    if 'ACC_NUM-SESSION-ID' in df.columns:
        df['FILE'] = df['ACC_NUM-SESSION-ID']
        print(f"✅ Created FILE column from ACC_NUM-SESSION-ID")
    else:
        print("❌ ACC_NUM-SESSION-ID column not found")
        return None
    
    # 2. Create GENDER column from SEX and filter
    if 'SEX' in df.columns:
        print(f"Original SEX distribution:")
        print(df['SEX'].value_counts())
        
        # Convert to lowercase and rename
        df['GENDER'] = df['SEX'].str.lower()
        
        # Filter out Unknown values
        initial_count = len(df)
        df = df[df['GENDER'].isin(['male', 'female'])].copy()
        filtered_count = len(df)
        
        print(f"✅ Created GENDER column, filtered {initial_count - filtered_count} Unknown records")
        print(f"Final GENDER distribution:")
        print(df['GENDER'].value_counts())
    else:
        print("❌ SEX column not found")
        return None
    
    return df

def process_hcc_codes(df):
    """Process HCC code columns as binary variables."""
    print("\n🏥 Processing HCC code columns...")
    
    hcc_cols = ['HCC_12', 'HCC_18', 'HCC_19', 'HCC_22', 'HCC_48', 'HCC_85', 'HCC_96', 'HCC_108', 'HCC_111']
    
    # Check for non-binary values
    print("📋 Examining HCC code value distributions:")
    non_binary_found = False
    
    for col in hcc_cols:
        if col in df.columns:
            unique_vals = sorted(df[col].unique())
            max_val = df[col].max()
            print(f"{col}: range 0-{max_val}, unique values: {len(unique_vals)}")
            
            if max_val > 1:
                non_binary_found = True
                print(f"  ⚠️  Non-binary values found: {unique_vals}")
        else:
            print(f"❌ {col} not found in dataset")
    
    if non_binary_found:
        print("\n🔍 HCC CODES ANALYSIS:")
        print("The HCC columns contain values > 1, indicating they represent counts/severity rather than binary presence.")
        print("Converting to binary: 0 = 'ABSENT', >0 = 'PRESENT'")
    
    # Process HCC columns
    processed_hcc_cols = []
    for col in hcc_cols:
        if col in df.columns:
            # Remove underscore and create new column name
            new_col_name = col.replace('_', '')  # HCC_12 -> HCC12
            
            # Convert to binary PRESENT/ABSENT
            df[new_col_name] = df[col].apply(lambda x: 'PRESENT' if x > 0 else 'ABSENT')
            processed_hcc_cols.append(new_col_name)
            
            # Show distribution
            distribution = df[new_col_name].value_counts()
            print(f"✅ {new_col_name}: {distribution['PRESENT']} PRESENT, {distribution['ABSENT']} ABSENT")
    
    print(f"✅ Processed {len(processed_hcc_cols)} HCC code columns")
    return df, processed_hcc_cols

def process_mortality(df):
    """Process mortality variable."""
    print("\n💀 Processing mortality variable...")
    
    if 'DEATH_DATE' in df.columns:
        # Create binary mortality variable
        df['MORTALITY'] = df['DEATH_DATE'].apply(lambda x: 'PRESENT' if pd.notna(x) else 'ABSENT')
        
        distribution = df['MORTALITY'].value_counts()
        print(f"✅ MORTALITY: {distribution['PRESENT']} PRESENT, {distribution['ABSENT']} ABSENT")
        print(f"   Mortality rate: {distribution['PRESENT'] / len(df) * 100:.2f}%")
    else:
        print("❌ DEATH_DATE column not found")
        return None
    
    return df

def process_diagnosis_variables(df):
    """Process diagnosis variables as binary."""
    print("\n🩺 Processing diagnosis variables...")
    
    diagnosis_mapping = {
        'Type_2_Diabetes_DX_Code': 'TYPE2DIABETES',
        'essential_HTN_DX_Code': 'ESSENTIALHTN', 
        'MI_DX_Code': 'MI',
        'Heart_failure_DX_Code': 'HEARTFAILURE'
    }
    
    processed_dx_cols = []
    for original_col, new_col in diagnosis_mapping.items():
        if original_col in df.columns:
            # Convert to binary PRESENT/ABSENT
            df[new_col] = df[original_col].apply(lambda x: 'PRESENT' if pd.notna(x) else 'ABSENT')
            processed_dx_cols.append(new_col)
            
            distribution = df[new_col].value_counts()
            print(f"✅ {new_col}: {distribution['PRESENT']} PRESENT, {distribution['ABSENT']} ABSENT")
        else:
            print(f"❌ {original_col} not found")
    
    print(f"✅ Processed {len(processed_dx_cols)} diagnosis columns")
    return df, processed_dx_cols

def process_calcium_scoring(df):
    """Process calcium scoring variables."""
    print("\n🦴 Processing calcium scoring variables...")
    
    if 'CalciumScoring_AbdominalAgatston' not in df.columns:
        print("❌ CalciumScoring_AbdominalAgatston column not found")
        return None
    
    # 1. Continuous variable (rename to all caps)
    df['CALCIUMSCORING_ABDOMINALAGATSTON'] = df['CalciumScoring_AbdominalAgatston']
    
    # Show statistics
    stats = df['CALCIUMSCORING_ABDOMINALAGATSTON'].describe()
    print(f"📊 Calcium scoring statistics:")
    print(f"   Range: {stats['min']:.1f} - {stats['max']:.1f}")
    print(f"   Mean: {stats['mean']:.1f}, Median: {stats['50%']:.1f}")
    
    # 2. Binary variable (>1000 = PRESENT)
    df['CALCIUMSCORING_ABDOMINALAGATSTON_BINARY'] = df['CALCIUMSCORING_ABDOMINALAGATSTON'].apply(
        lambda x: 'PRESENT' if x > 1000 else 'ABSENT'
    )
    
    binary_dist = df['CALCIUMSCORING_ABDOMINALAGATSTON_BINARY'].value_counts()
    print(f"✅ Binary calcium scoring (>1000): {binary_dist['PRESENT']} PRESENT, {binary_dist['ABSENT']} ABSENT")
    
    # 3. Multiclass variable
    def calcium_multiclass(x):
        if x == 0:
            return 'ABSENT'
        elif 0 < x <= 1000:
            return 'LOW'
        elif 1000 < x <= 3000:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    df['CALCIUMSCORING_ABDOMINALAGATSTON_MULTICLASS'] = df['CALCIUMSCORING_ABDOMINALAGATSTON'].apply(calcium_multiclass)
    
    multiclass_dist = df['CALCIUMSCORING_ABDOMINALAGATSTON_MULTICLASS'].value_counts()
    print(f"✅ Multiclass calcium scoring distribution:")
    for category, count in multiclass_dist.items():
        print(f"   {category}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def process_biomarker_variables(df):
    """Process biomarker variables."""
    print("\n🔬 Processing biomarker variables...")
    
    biomarker_mapping = {
        'KidneyValues_KidneyMedianHU': 'KIDNEYVALUES_KIDNEYMEDIANHU',
        'SpleenValues_SpleenVolume': 'SPLEENVALUES_SPLEENVOLUME'
    }
    
    processed_biomarker_cols = []
    for original_col, new_col in biomarker_mapping.items():
        if original_col in df.columns:
            df[new_col] = df[original_col]
            processed_biomarker_cols.append(new_col)
            
            # Show statistics
            stats = df[new_col].describe()
            non_null_count = df[new_col].notna().sum()
            print(f"✅ {new_col}: {non_null_count} valid values, range {stats['min']:.1f}-{stats['max']:.1f}")
        else:
            print(f"❌ {original_col} not found")
    
    return df, processed_biomarker_cols

def process_age_variable(df):
    """Process age variable."""
    print("\n👤 Processing age variable...")
    
    if 'AGE' in df.columns:
        # Convert to integer
        df['AGE'] = df['AGE'].astype(int)
        
        age_stats = df['AGE'].describe()
        print(f"✅ AGE: range {int(age_stats['min'])}-{int(age_stats['max'])} years")
        print(f"   Mean: {age_stats['mean']:.1f}, Median: {int(age_stats['50%'])}")
    else:
        print("❌ AGE column not found")
        return None
    
    return df

def create_hcc_comorbidity_severity(df, hcc_cols):
    """Create HCC comorbidity severity variable."""
    print("\n🏥 Creating HCC comorbidity severity variable...")
    
    # Count number of PRESENT HCC conditions for each patient
    hcc_present_counts = []
    for _, row in df.iterrows():
        present_count = sum(1 for col in hcc_cols if row[col] == 'PRESENT')
        hcc_present_counts.append(present_count)
    
    df['HCC_COMORBIDITY_SEVERITY'] = hcc_present_counts
    
    # Create severity categories as defined in EDA
    def severity_category(count):
        if count == 0:
            return 'NONE'
        elif 1 <= count <= 2:
            return 'LOW'
        elif 3 <= count <= 5:
            return 'MODERATE'
        else:
            return 'HIGH'
    
    df['HCC_COMORBIDITY_SEVERITY_CATEGORY'] = df['HCC_COMORBIDITY_SEVERITY'].apply(severity_category)
    
    # Show distribution
    severity_dist = df['HCC_COMORBIDITY_SEVERITY_CATEGORY'].value_counts()
    print(f"✅ HCC Comorbidity Severity distribution:")
    for category, count in severity_dist.items():
        print(f"   {category}: {count} ({count/len(df)*100:.1f}%)")
    
    severity_numeric_dist = df['HCC_COMORBIDITY_SEVERITY'].describe()
    print(f"📊 Numeric severity: mean {severity_numeric_dist['mean']:.1f}, max {int(severity_numeric_dist['max'])}")
    
    return df

def create_patient_splits(df):
    """Create train/val/test splits by patient ID."""
    print("\n📊 Creating patient-level train/val/test splits...")
    
    if 'PAT-ID' not in df.columns:
        print("❌ PAT-ID column not found")
        return None, None, None
    
    # Get unique patients
    unique_patients = df['PAT-ID'].unique()
    n_patients = len(unique_patients)
    
    print(f"📋 Total unique patients: {n_patients:,}")
    print(f"📋 Total records: {len(df):,}")
    print(f"📋 Average records per patient: {len(df)/n_patients:.2f}")
    
    # Split patients (not records) into train/val/test
    # First split: 70% train, 30% temp
    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.3, random_state=42
    )
    
    # Second split: 20% val, 10% test from the 30% temp
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.33333, random_state=42  # 0.33333 of 30% = 10% of total
    )
    
    print(f"📊 Patient splits:")
    print(f"   Train: {len(train_patients):,} patients ({len(train_patients)/n_patients*100:.1f}%)")
    print(f"   Val: {len(val_patients):,} patients ({len(val_patients)/n_patients*100:.1f}%)")
    print(f"   Test: {len(test_patients):,} patients ({len(test_patients)/n_patients*100:.1f}%)")
    
    # Create record splits based on patient assignments
    train_df = df[df['PAT-ID'].isin(train_patients)].copy()
    val_df = df[df['PAT-ID'].isin(val_patients)].copy()
    test_df = df[df['PAT-ID'].isin(test_patients)].copy()
    
    print(f"📊 Record splits:")
    print(f"   Train: {len(train_df):,} records ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Val: {len(val_df):,} records ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test: {len(test_df):,} records ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify no patient overlap
    train_patients_set = set(train_df['PAT-ID'].unique())
    val_patients_set = set(val_df['PAT-ID'].unique())
    test_patients_set = set(test_df['PAT-ID'].unique())
    
    overlap_train_val = train_patients_set.intersection(val_patients_set)
    overlap_train_test = train_patients_set.intersection(test_patients_set)
    overlap_val_test = val_patients_set.intersection(test_patients_set)
    
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print("✅ No patient overlap between splits confirmed")
    else:
        print(f"❌ Patient overlap detected: train-val={len(overlap_train_val)}, train-test={len(overlap_train_test)}, val-test={len(overlap_val_test)}")
    
    return train_df, val_df, test_df

def select_final_columns(df, hcc_cols, dx_cols, biomarker_cols):
    """Select and order final columns for the modeling dataset."""
    print("\n📋 Selecting final columns for modeling dataset...")
    
    # Define final column order (excluding PAT-ID from output)
    final_columns = [
        # Identifiers
        'FILE',
        
        # Demographics
        'GENDER',
        'AGE',
        
        # Mortality
        'MORTALITY',
        
        # Diagnoses
        *dx_cols,
        
        # HCC Codes
        *hcc_cols,
        'HCC_COMORBIDITY_SEVERITY',
        'HCC_COMORBIDITY_SEVERITY_CATEGORY',
        
        # Calcium Scoring
        'CALCIUMSCORING_ABDOMINALAGATSTON',
        'CALCIUMSCORING_ABDOMINALAGATSTON_BINARY',
        'CALCIUMSCORING_ABDOMINALAGATSTON_MULTICLASS',
        
        # Biomarkers
        *biomarker_cols
    ]
    
    # Filter to only include columns that exist
    available_columns = [col for col in final_columns if col in df.columns]
    missing_columns = [col for col in final_columns if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️  Missing columns: {missing_columns}")
    
    print(f"✅ Selected {len(available_columns)} columns for final dataset")
    
    return df[available_columns].copy()

def save_datasets(train_df, val_df, test_df):
    """Save the processed datasets."""
    print("\n💾 Saving processed datasets...")
    
    # Create output directory
    output_dir = "../../datasets/full_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save datasets
    datasets = {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }
    
    saved_files = []
    for split_name, split_df in datasets.items():
        filename = f"modeling_dataset_{split_name}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        split_df.to_csv(filepath, index=False)
        saved_files.append(filepath)
        
        print(f"✅ {split_name.upper()} set: {len(split_df):,} records → {filename}")
    
    # Save summary
    summary_filename = f"modeling_dataset_summary_{timestamp}.txt"
    summary_path = os.path.join(output_dir, summary_filename)
    
    with open(summary_path, 'w') as f:
        f.write("MODELING DATASET CREATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: biomarkers_with_hcc_codes_20250909_062523.csv\n\n")
        
        f.write("DATASET SPLITS:\n")
        f.write(f"Train: {len(train_df):,} records ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)\n")
        f.write(f"Val: {len(val_df):,} records ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)\n")
        f.write(f"Test: {len(test_df):,} records ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)\n")
        f.write(f"Total: {len(train_df)+len(val_df)+len(test_df):,} records\n\n")
        
        f.write("COLUMNS INCLUDED:\n")
        for i, col in enumerate(train_df.columns, 1):
            f.write(f"{i:2d}. {col}\n")
        
        f.write(f"\nPatient-level splits ensure no data leakage between train/val/test sets.\n")
    
    print(f"📋 Summary saved: {summary_filename}")
    print(f"📁 All files saved in: {output_dir}")
    
    return saved_files

def main():
    """Main execution function."""
    print("=" * 80)
    print("MODELING DATASET CREATION WITH TRAIN/VAL/TEST SPLITS")
    print("=" * 80)
    print(f"Processing started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load and examine data
    df = load_and_examine_data()
    if df is None:
        return
    
    # 2. Process identifier and gender
    df = process_identifier_and_gender(df)
    if df is None:
        return
    
    # 3. Process HCC codes
    df, hcc_cols = process_hcc_codes(df)
    
    # 4. Process mortality
    df = process_mortality(df)
    if df is None:
        return
    
    # 5. Process diagnosis variables
    df, dx_cols = process_diagnosis_variables(df)
    
    # 6. Process calcium scoring
    df = process_calcium_scoring(df)
    if df is None:
        return
    
    # 7. Process biomarker variables
    df, biomarker_cols = process_biomarker_variables(df)
    
    # 8. Process age
    df = process_age_variable(df)
    if df is None:
        return
    
    # 9. Create HCC comorbidity severity
    df = create_hcc_comorbidity_severity(df, hcc_cols)
    
    # 10. Create patient splits (before selecting final columns, need PAT-ID for splitting)
    train_df, val_df, test_df = create_patient_splits(df)
    if train_df is None:
        return
    
    # 11. Select final columns (excluding PAT-ID) for each split
    train_df_final = select_final_columns(train_df, hcc_cols, dx_cols, biomarker_cols)
    val_df_final = select_final_columns(val_df, hcc_cols, dx_cols, biomarker_cols)
    test_df_final = select_final_columns(test_df, hcc_cols, dx_cols, biomarker_cols)
    
    # 12. Save datasets
    saved_files = save_datasets(train_df_final, val_df_final, test_df_final)
    
    print("\n" + "=" * 80)
    print("✅ MODELING DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📊 Final dataset summary:")
    print(f"   Total records: {len(train_df_final) + len(val_df_final) + len(test_df_final):,}")
    print(f"   Total columns: {len(train_df_final.columns)}")
    print(f"   Train/Val/Test: {len(train_df_final):,}/{len(val_df_final):,}/{len(test_df_final):,}")
    print(f"🎯 Ready for multi-task modeling!")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
