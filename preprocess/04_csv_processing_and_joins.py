#!/usr/bin/env python3
"""
CSV Processing and Joins
========================

Performs the following operations on the biomarkers CSV:
1. Left join to add PNG filename column based on matched pairs
2. Left join with oscar_master_cohort-lookup.csv to get PAT-ID
3. Validation and statistics reporting

Author: AI Assistant  
Date: 2025-09-04
"""

import os
import pandas as pd
from datetime import datetime
import numpy as np


def load_data():
    """Load all required CSV files."""
    print("🚀 CSV PROCESSING AND JOINS")
    print("=" * 60)
    
    # Define file paths
    biomarkers_csv = os.path.join(os.path.dirname(__file__), 
                                  '../../datasets/full_data/2025_08_31_Biomarkers_Outcomes_Joined_Fixed.csv')
    matched_pairs_csv = os.path.join(os.path.dirname(__file__), 
                                     'matching_results/matched_pairs_20250904_051334.csv')
    lookup_csv = os.path.join(os.path.dirname(__file__), 
                              '../../datasets/full_data/oscar_master_cohort-lookup.csv')
    
    print("📊 Loading CSV files...")
    
    # Load biomarkers data
    try:
        biomarkers_df = pd.read_csv(biomarkers_csv, low_memory=False)
        print(f"✅ Biomarkers CSV: {len(biomarkers_df):,} rows, {len(biomarkers_df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading biomarkers CSV: {e}")
        return None, None, None
    
    # Load matched pairs
    try:
        matched_pairs_df = pd.read_csv(matched_pairs_csv)
        print(f"✅ Matched pairs CSV: {len(matched_pairs_df):,} rows, {len(matched_pairs_df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading matched pairs CSV: {e}")
        return None, None, None
    
    # Load lookup data
    try:
        lookup_df = pd.read_csv(lookup_csv, low_memory=False)
        print(f"✅ Lookup CSV: {len(lookup_df):,} rows, {len(lookup_df.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading lookup CSV: {e}")
        return None, None, None
    
    return biomarkers_df, matched_pairs_df, lookup_df


def add_png_filename_column(biomarkers_df, matched_pairs_df):
    """Add PNG filename column via left join."""
    print("\n🔗 STEP 1: Adding PNG filename column...")
    
    # Check the column names in matched_pairs
    print(f"Matched pairs columns: {list(matched_pairs_df.columns)}")
    
    # Perform left join on ACC_NUM-SESSION-ID
    result_df = biomarkers_df.merge(
        matched_pairs_df, 
        left_on='ACC_NUM-SESSION-ID', 
        right_on='acc_num_session_id', 
        how='left'
    )
    
    # Drop the duplicate column from matched_pairs
    if 'acc_num_session_id' in result_df.columns:
        result_df = result_df.drop('acc_num_session_id', axis=1)
    
    # Count PNG matches
    png_matched_count = result_df['png_filename'].notna().sum()
    total_rows = len(result_df)
    
    print(f"📊 Results after PNG filename join:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Rows with PNG filename: {png_matched_count:,}")
    print(f"   Rows without PNG filename: {total_rows - png_matched_count:,}")
    print(f"   PNG match rate: {png_matched_count/total_rows:.2%}")
    
    return result_df, png_matched_count


def add_pat_id_column(result_df, lookup_df):
    """Add PAT-ID column via left join with oscar_master_cohort-lookup.csv."""
    print("\n🔗 STEP 2: Adding PAT-ID column from lookup...")
    
    # Check the column names in lookup
    print(f"Lookup CSV columns: {list(lookup_df.columns)}")
    
    # Check if ACC_NUM column exists in lookup
    if 'ACC_NUM' not in lookup_df.columns:
        print("❌ Error: 'ACC_NUM' column not found in lookup CSV")
        print(f"Available columns: {list(lookup_df.columns)}")
        return None, 0
    
    # Check if PAT-ID column exists in lookup  
    pat_id_col = None
    for col in lookup_df.columns:
        if 'PAT' in col.upper() and 'ID' in col.upper():
            pat_id_col = col
            break
    
    if pat_id_col is None:
        print("❌ Error: PAT-ID column not found in lookup CSV")
        print(f"Available columns: {list(lookup_df.columns)}")
        return None, 0
    
    print(f"Using PAT-ID column: '{pat_id_col}'")
    
    # Extract ACC_NUM from ACC_NUM-SESSION-ID (first 6 digits)
    result_df['ACC_NUM_EXTRACTED'] = result_df['ACC_NUM-SESSION-ID'].str[:6]
    
    # Perform left join
    final_df = result_df.merge(
        lookup_df[['ACC_NUM', pat_id_col]], 
        left_on='ACC_NUM-SESSION-ID', 
        right_on='ACC_NUM', 
        how='left'
    )
    
    # Drop temporary columns
    final_df = final_df.drop(['ACC_NUM_EXTRACTED', 'ACC_NUM'], axis=1)
    
    # Count PAT-ID matches
    pat_id_matched_count = final_df[pat_id_col].notna().sum()
    total_rows = len(final_df)
    
    print(f"📊 Results after PAT-ID join:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Rows with PAT-ID: {pat_id_matched_count:,}")
    print(f"   Rows without PAT-ID: {total_rows - pat_id_matched_count:,}")
    print(f"   PAT-ID match rate: {pat_id_matched_count/total_rows:.2%}")
    
    return final_df, pat_id_matched_count


def validate_and_report(final_df, png_matched_count, pat_id_matched_count):
    """Validate results and generate comprehensive report."""
    print("\n📊 VALIDATION AND FINAL STATISTICS")
    print("=" * 60)
    
    total_rows = len(final_df)
    
    print(f"📋 FINAL DATASET SUMMARY:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Total columns: {len(final_df.columns)}")
    print()
    
    print(f"🖼️  PNG FILENAME MATCHING:")
    print(f"   Expected PNG matches: 48,770")
    print(f"   Actual PNG matches: {png_matched_count:,}")
    print(f"   ✅ Match validation: {'PASSED' if png_matched_count == 48770 else 'FAILED'}")
    print()
    
    print(f"🆔 PAT-ID MATCHING:")
    print(f"   Rows with PAT-ID: {pat_id_matched_count:,}")
    print(f"   PAT-ID match rate: {pat_id_matched_count/total_rows:.2%}")
    print()
    
    # Cross-tabulation analysis
    if 'png_filename' in final_df.columns:
        pat_id_col = None
        for col in final_df.columns:
            if 'PAT' in col.upper() and 'ID' in col.upper():
                pat_id_col = col
                break
        
        if pat_id_col:
            print(f"🔍 CROSS-TABULATION ANALYSIS:")
            has_png = final_df['png_filename'].notna()
            has_pat_id = final_df[pat_id_col].notna()
            
            both_matched = (has_png & has_pat_id).sum()
            png_only = (has_png & ~has_pat_id).sum()
            pat_id_only = (~has_png & has_pat_id).sum()
            neither = (~has_png & ~has_pat_id).sum()
            
            print(f"   Both PNG and PAT-ID: {both_matched:,}")
            print(f"   PNG only (no PAT-ID): {png_only:,}")
            print(f"   PAT-ID only (no PNG): {pat_id_only:,}")
            print(f"   Neither PNG nor PAT-ID: {neither:,}")
            print()
            
            if both_matched > 0:
                print(f"✅ Complete records (PNG + PAT-ID): {both_matched:,} ({both_matched/total_rows:.2%})")
    
    return final_df


def save_results(final_df):
    """Save the final processed dataset."""
    print("\n💾 SAVING RESULTS...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"../../datasets/full_data/biomarkers_with_png_and_patid_{timestamp}.csv"
    
    try:
        final_df.to_csv(output_file, index=False)
        print(f"✅ Final dataset saved: {output_file}")
        print(f"   Rows: {len(final_df):,}")
        print(f"   Columns: {len(final_df.columns)}")
        print(f"   File size: {os.path.getsize(output_file) / (1024*1024):.1f} MB")
    except Exception as e:
        print(f"❌ Error saving file: {e}")


def main():
    """Main execution function."""
    # Load data
    biomarkers_df, matched_pairs_df, lookup_df = load_data()
    if biomarkers_df is None:
        return
    
    # Step 1: Add PNG filename column
    result_df, png_matched_count = add_png_filename_column(biomarkers_df, matched_pairs_df)
    
    # Step 2: Add PAT-ID column
    final_df, pat_id_matched_count = add_pat_id_column(result_df, lookup_df)
    if final_df is None:
        return
    
    # Step 3: Validate and report
    final_df = validate_and_report(final_df, png_matched_count, pat_id_matched_count)
    
    # Step 4: Save results
    save_results(final_df)
    
    print(f"\n🎉 SUCCESS: CSV processing and joins completed!")


if __name__ == "__main__":
    main()
