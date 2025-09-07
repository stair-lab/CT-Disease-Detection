#!/usr/bin/env python3
"""
Filter Complete Records
=======================

Filters the processed biomarkers CSV to keep only records that have both:
1. PNG filename (image available)
2. PAT-ID (patient ID available)

This creates a clean dataset ready for machine learning analysis.

Author: AI Assistant
Date: 2025-09-04
"""

import os
import pandas as pd
from datetime import datetime


def filter_complete_records():
    """Filter to keep only records with both PNG filename and PAT-ID."""
    print("🚀 FILTERING COMPLETE RECORDS")
    print("=" * 60)
    
    # Define input and output paths
    input_file = os.path.join(os.path.dirname(__file__), 
                              '../../datasets/full_data/biomarkers_with_png_and_patid_20250904_054446.csv')
    output_file = os.path.join(os.path.dirname(__file__), 
                               '../../datasets/full_data/2025_09_04_Biomarkers_Outcomes_final.csv')
    
    print(f"📊 Input file: {os.path.basename(input_file)}")
    print(f"📁 Output file: {os.path.basename(output_file)}")
    
    # Load the processed dataset
    try:
        df = pd.read_csv(input_file, low_memory=False)
        print(f"✅ Loaded dataset: {len(df):,} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"❌ Error: Input file not found at {input_file}")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Check required columns exist
    required_cols = ['png_filename', 'PAT-ID']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing required columns: {missing_cols}")
        return
    
    print(f"\n🔍 INITIAL DATA ANALYSIS:")
    print(f"   Total records: {len(df):,}")
    
    # Analyze completeness
    has_png = df['png_filename'].notna()
    has_pat_id = df['PAT-ID'].notna()
    
    png_count = has_png.sum()
    pat_id_count = has_pat_id.sum()
    complete_count = (has_png & has_pat_id).sum()
    
    print(f"   Records with PNG: {png_count:,} ({png_count/len(df):.2%})")
    print(f"   Records with PAT-ID: {pat_id_count:,} ({pat_id_count/len(df):.2%})")
    print(f"   Complete records (both): {complete_count:,} ({complete_count/len(df):.2%})")
    
    # Filter to keep only complete records
    print(f"\n🔗 FILTERING COMPLETE RECORDS...")
    filtered_df = df[has_png & has_pat_id].copy()
    
    print(f"✅ Filtered dataset:")
    print(f"   Records kept: {len(filtered_df):,}")
    print(f"   Records removed: {len(df) - len(filtered_df):,}")
    print(f"   Retention rate: {len(filtered_df)/len(df):.2%}")
    
    # Verify no missing values in key columns
    png_missing = filtered_df['png_filename'].isnull().sum()
    pat_id_missing = filtered_df['PAT-ID'].isnull().sum()
    
    print(f"\n🔍 QUALITY CHECK:")
    print(f"   PNG filename missing: {png_missing}")
    print(f"   PAT-ID missing: {pat_id_missing}")
    
    if png_missing == 0 and pat_id_missing == 0:
        print("   ✅ All records have both PNG filename and PAT-ID")
    else:
        print("   ❌ Warning: Some records still missing required data")
    
    # Save the filtered dataset
    print(f"\n💾 SAVING FILTERED DATASET...")
    try:
        filtered_df.to_csv(output_file, index=False)
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"✅ Saved: {output_file}")
        print(f"   Rows: {len(filtered_df):,}")
        print(f"   Columns: {len(filtered_df.columns)}")
        print(f"   File size: {file_size_mb:.1f} MB")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return
    
    # Generate summary statistics
    print(f"\n📊 FINAL DATASET SUMMARY:")
    print(f"   Dataset name: 2025_09_04_Biomarkers_Outcomes_final.csv")
    print(f"   Total records: {len(filtered_df):,}")
    print(f"   Total features: {len(filtered_df.columns)}")
    print(f"   PNG files available: {len(filtered_df):,}")
    print(f"   Patient IDs available: {len(filtered_df):,}")
    print(f"   Ready for ML analysis: ✅")
    
    # Show sample of key columns
    print(f"\n📋 SAMPLE DATA:")
    sample_cols = ['ACC_NUM-SESSION-ID', 'PAT_ID_MASKED', 'png_filename', 'PAT-ID']
    available_cols = [col for col in sample_cols if col in filtered_df.columns]
    
    if available_cols:
        print(filtered_df[available_cols].head().to_string(index=False))
    
    print(f"\n🎉 SUCCESS: Complete records dataset created!")
    print(f"📁 Location: datasets/full_data/2025_09_04_Biomarkers_Outcomes_final.csv")


def main():
    """Main execution function."""
    filter_complete_records()


if __name__ == "__main__":
    main()

