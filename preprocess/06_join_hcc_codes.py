#!/usr/bin/env python3
"""
Join HCC Codes with Biomarkers Data
===================================

This script joins the HCC codes from Oscar-HCC-codes.xlsx with the biomarkers data
from 2025_09_04_Biomarkers_Outcomes_final.csv using ACC_NUM as the join key.

The script:
1. Reads the biomarkers CSV file
2. Reads the HCC codes Excel file
3. Extracts ACC_NUM from ACC_NUM-SESSION-ID in the CSV file
4. Joins the data on ACC_NUM
5. Adds "HCC_" prefix to numeric column names from the Excel file
6. Saves the joined dataset

Author: AI Assistant
Date: 2025-01-27
"""

import os
import pandas as pd
from datetime import datetime
import numpy as np


def extract_acc_num_from_session_id(session_id):
    """Extract ACC_NUM from ACC_NUM-SESSION-ID format."""
    # The full ACC_NUM-SESSION-ID matches the ACC_NUM in the Excel file
    if pd.isna(session_id):
        return None
    return str(session_id)


def load_data():
    """Load biomarkers CSV and HCC codes Excel files."""
    print("🚀 HCC CODES JOIN PROCESSING")
    print("=" * 60)
    
    # Define file paths
    biomarkers_csv = os.path.join(os.path.dirname(__file__), 
                                  '../../datasets/full_data/2025_09_04_Biomarkers_Outcomes_final.csv')
    hcc_excel = os.path.join(os.path.dirname(__file__), 
                            '../../datasets/full_data/Oscar-HCC-codes.xlsx')
    
    print("📊 Loading data files...")
    
    # Load biomarkers data
    try:
        biomarkers_df = pd.read_csv(biomarkers_csv, low_memory=False)
        print(f"✅ Biomarkers CSV: {len(biomarkers_df):,} rows, {len(biomarkers_df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading biomarkers CSV: {e}")
        return None, None
    
    # Load HCC codes data
    try:
        hcc_df = pd.read_excel(hcc_excel)
        print(f"✅ HCC Excel: {len(hcc_df):,} rows, {len(hcc_df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading HCC Excel: {e}")
        return None, None
    
    return biomarkers_df, hcc_df


def preprocess_data(biomarkers_df, hcc_df):
    """Preprocess data for joining."""
    print("\n🔧 Preprocessing data...")
    
    # Use full ACC_NUM-SESSION-ID as it matches the ACC_NUM in Excel file
    print("📋 Using full ACC_NUM-SESSION-ID as join key...")
    biomarkers_df['ACC_NUM_EXTRACTED'] = biomarkers_df['ACC_NUM-SESSION-ID'].apply(extract_acc_num_from_session_id)
    
    # Show some examples
    print("Sample ACC_NUM extractions:")
    sample_extractions = biomarkers_df[['ACC_NUM-SESSION-ID', 'ACC_NUM_EXTRACTED']].head()
    for _, row in sample_extractions.iterrows():
        print(f"  {row['ACC_NUM-SESSION-ID']} -> {row['ACC_NUM_EXTRACTED']}")
    
    # Identify numeric columns in HCC data to add prefix
    numeric_cols = [col for col in hcc_df.columns if str(col).isdigit()]
    print(f"\n🏷️  Found {len(numeric_cols)} numeric HCC code columns to prefix")
    print(f"First 10 numeric columns: {numeric_cols[:10]}")
    
    # Create a mapping for renaming columns
    column_mapping = {col: f'HCC_{col}' for col in numeric_cols}
    
    # Rename numeric columns in HCC dataframe
    hcc_df_renamed = hcc_df.rename(columns=column_mapping)
    print(f"✅ Renamed {len(column_mapping)} columns with 'HCC_' prefix")
    
    return biomarkers_df, hcc_df_renamed, numeric_cols


def perform_join(biomarkers_df, hcc_df_renamed):
    """Perform the join operation."""
    print("\n🔗 Performing join operation...")
    
    # Check for duplicates in join keys
    biomarkers_acc_counts = biomarkers_df['ACC_NUM_EXTRACTED'].value_counts()
    hcc_acc_counts = hcc_df_renamed['ACC_NUM'].value_counts()
    
    print(f"📊 Join key statistics:")
    print(f"  Biomarkers unique ACC_NUM: {biomarkers_df['ACC_NUM_EXTRACTED'].nunique():,}")
    print(f"  HCC unique ACC_NUM: {hcc_df_renamed['ACC_NUM'].nunique():,}")
    print(f"  Biomarkers duplicated ACC_NUM: {(biomarkers_acc_counts > 1).sum():,}")
    print(f"  HCC duplicated ACC_NUM: {(hcc_acc_counts > 1).sum():,}")
    
    # Find overlapping ACC_NUMs
    biomarkers_acc_set = set(biomarkers_df['ACC_NUM_EXTRACTED'].dropna())
    hcc_acc_set = set(hcc_df_renamed['ACC_NUM'].dropna())
    overlap = biomarkers_acc_set.intersection(hcc_acc_set)
    
    print(f"  Overlapping ACC_NUMs: {len(overlap):,}")
    print(f"  Biomarkers only: {len(biomarkers_acc_set - hcc_acc_set):,}")
    print(f"  HCC only: {len(hcc_acc_set - biomarkers_acc_set):,}")
    
    # Perform left join
    print("\n🔄 Executing left join...")
    joined_df = pd.merge(
        biomarkers_df, 
        hcc_df_renamed, 
        left_on='ACC_NUM_EXTRACTED', 
        right_on='ACC_NUM', 
        how='left',
        suffixes=('', '_hcc')
    )
    
    print(f"✅ Join completed:")
    print(f"  Final dataset: {len(joined_df):,} rows, {len(joined_df.columns)} columns")
    
    # Check if we have the HCC ACC_NUM column (might be named differently due to suffix handling)
    hcc_acc_col = None
    for col in joined_df.columns:
        if 'ACC_NUM' in col and col != 'ACC_NUM_EXTRACTED' and col != 'ACC_NUM-SESSION-ID':
            hcc_acc_col = col
            break
    
    if hcc_acc_col:
        print(f"  Records with HCC data: {joined_df[hcc_acc_col].notna().sum():,}")
        print(f"  Records without HCC data: {joined_df[hcc_acc_col].isna().sum():,}")
    else:
        # If no separate HCC ACC_NUM column, check for HCC columns directly
        hcc_cols = [col for col in joined_df.columns if col.startswith('HCC_')]
        if hcc_cols:
            # Use first HCC column to determine coverage
            sample_hcc_col = hcc_cols[0]
            print(f"  Records with HCC data: {joined_df[sample_hcc_col].notna().sum():,}")
            print(f"  Records without HCC data: {joined_df[sample_hcc_col].isna().sum():,}")
        else:
            print("  Warning: No HCC columns found in joined dataset")
    
    return joined_df


def analyze_joined_data(joined_df, original_numeric_cols):
    """Analyze the joined dataset."""
    print("\n📈 Analyzing joined dataset...")
    
    # Check HCC code coverage
    hcc_cols = [f'HCC_{col}' for col in original_numeric_cols]
    
    # Calculate statistics for HCC columns
    hcc_stats = []
    for col in hcc_cols[:10]:  # Show stats for first 10 HCC columns
        if col in joined_df.columns:
            non_null_count = joined_df[col].notna().sum()
            positive_count = (joined_df[col] == 1).sum()
            hcc_stats.append({
                'HCC_Code': col,
                'Non_Null_Records': non_null_count,
                'Positive_Cases': positive_count,
                'Positive_Rate': f"{positive_count/non_null_count*100:.2f}%" if non_null_count > 0 else "0%"
            })
    
    if hcc_stats:
        hcc_stats_df = pd.DataFrame(hcc_stats)
        print("\n📊 HCC Code Statistics (first 10 codes):")
        print(hcc_stats_df.to_string(index=False))
    
    # Overall HCC coverage
    total_hcc_cols = len([col for col in joined_df.columns if col.startswith('HCC_')])
    print(f"\n🏷️  Total HCC columns added: {total_hcc_cols}")
    
    # Check for any missing demographic data from HCC file
    if 'SEX_hcc' in joined_df.columns:
        sex_coverage = joined_df['SEX_hcc'].notna().sum()
        print(f"📋 Records with HCC demographic data: {sex_coverage:,}")


def save_results(joined_df):
    """Save the joined dataset."""
    print("\n💾 Saving results...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '../../datasets/processed_data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"biomarkers_with_hcc_codes_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save the joined dataset
    try:
        joined_df.to_csv(output_path, index=False)
        print(f"✅ Joined dataset saved: {output_path}")
        print(f"📊 Final dataset: {len(joined_df):,} rows, {len(joined_df.columns)} columns")
        
        # Save a summary file
        summary_filename = f"hcc_join_summary_{timestamp}.txt"
        summary_path = os.path.join(output_dir, summary_filename)
        
        with open(summary_path, 'w') as f:
            f.write(f"HCC Codes Join Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input biomarkers file: 2025_09_04_Biomarkers_Outcomes_final.csv\n")
            f.write(f"Input HCC file: Oscar-HCC-codes.xlsx\n")
            f.write(f"Output file: {output_filename}\n")
            f.write(f"Final dataset: {len(joined_df):,} rows, {len(joined_df.columns)} columns\n")
            # Find HCC coverage using HCC columns
            hcc_cols = [col for col in joined_df.columns if col.startswith('HCC_')]
            if hcc_cols:
                sample_hcc_col = hcc_cols[0]
                hcc_records = joined_df[sample_hcc_col].notna().sum()
                no_hcc_records = joined_df[sample_hcc_col].isna().sum()
                f.write(f"Records with HCC data: {hcc_records:,}\n")
                f.write(f"Records without HCC data: {no_hcc_records:,}\n")
            else:
                f.write("Records with HCC data: Unknown\n")
                f.write("Records without HCC data: Unknown\n")
            f.write(f"HCC columns added: {len([col for col in joined_df.columns if col.startswith('HCC_')])}\n")
        
        print(f"📋 Summary saved: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def main():
    """Main execution function."""
    # Load data
    biomarkers_df, hcc_df = load_data()
    if biomarkers_df is None or hcc_df is None:
        return
    
    # Preprocess data
    biomarkers_df, hcc_df_renamed, original_numeric_cols = preprocess_data(biomarkers_df, hcc_df)
    
    # Perform join
    joined_df = perform_join(biomarkers_df, hcc_df_renamed)
    
    # Analyze results
    analyze_joined_data(joined_df, original_numeric_cols)
    
    # Save results
    save_results(joined_df)
    
    print("\n🎉 HCC codes join processing completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
