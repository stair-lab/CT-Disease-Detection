#!/usr/bin/env python3
"""
Age Anonymization Script
========================

This script reads train.csv, val.csv, and test.csv files and converts
any AGE values >= 90 to "90+" for HIPAA compliance.

Author: AI Assistant
Date: 2025-09-21
"""

import os
import pandas as pd
from datetime import datetime

def anonymize_age_column(df, filename):
    """Anonymize AGE column by converting values >= 90 to '90+'."""
    print(f"\n📊 Processing {filename}...")
    
    if 'AGE' not in df.columns:
        print(f"❌ AGE column not found in {filename}")
        return df
    
    # Count records with age >= 90 before anonymization
    ages_90_plus = (df['AGE'] >= 90).sum()
    total_records = len(df)
    
    print(f"   Total records: {total_records:,}")
    print(f"   Records with AGE >= 90: {ages_90_plus:,}")
    
    if ages_90_plus > 0:
        # Convert ages >= 90 to "90+"
        df['AGE'] = df['AGE'].apply(lambda x: "90+" if x >= 90 else x)
        print(f"   ✅ Converted {ages_90_plus:,} ages >= 90 to '90+'")
        
        # Show age distribution after anonymization
        age_distribution = df['AGE'].value_counts()
        print(f"   📋 Age distribution after anonymization:")
        for age, count in age_distribution.items():
            if str(age) == "90+" or (isinstance(age, int) and age >= 85):
                print(f"      Age {age}: {count:,} records")
    else:
        print(f"   ✅ No ages >= 90 found in {filename}")
    
    return df

def verify_only_age_changed(df_original, df_anonymized, filename):
    """Verify that only the AGE column changed between original and anonymized data."""
    print(f"\n🔍 Verifying data integrity for {filename}...")
    
    # Check if both dataframes have the same shape
    if df_original.shape != df_anonymized.shape:
        print(f"   ❌ Shape mismatch: {df_original.shape} vs {df_anonymized.shape}")
        return False
    
    # Check if both dataframes have the same columns
    if list(df_original.columns) != list(df_anonymized.columns):
        print(f"   ❌ Column mismatch detected")
        return False
    
    # Check each column for differences
    differences_found = []
    for col in df_original.columns:
        if col == 'AGE':
            # For AGE column, check if only values >= 90 changed to "90+"
            original_ages = df_original[col]
            anonymized_ages = df_anonymized[col]
            
            # Find records where AGE changed
            age_changed = (original_ages != anonymized_ages)
            changed_count = age_changed.sum()
            
            if changed_count > 0:
                # Verify that all changes are from >= 90 to "90+"
                changed_indices = df_original[age_changed].index
                all_valid_changes = True
                
                for idx in changed_indices:
                    orig_age = original_ages.iloc[idx]
                    anon_age = anonymized_ages.iloc[idx]
                    
                    # Check if original age was >= 90 and anonymized age is "90+"
                    if not (orig_age >= 90 and anon_age == "90+"):
                        all_valid_changes = False
                        print(f"   ❌ Invalid AGE change at index {idx}: {orig_age} → {anon_age}")
                        break
                
                if all_valid_changes:
                    print(f"   ✅ AGE column: {changed_count:,} valid changes (>= 90 → '90+')")
                else:
                    print(f"   ❌ AGE column: Invalid changes detected")
                    return False
            else:
                print(f"   ✅ AGE column: No changes needed")
        else:
            # For non-AGE columns, check if they're identical
            if not df_original[col].equals(df_anonymized[col]):
                differences_found.append(col)
                print(f"   ❌ Column '{col}' has unexpected differences")
    
    if differences_found:
        print(f"   ❌ Data integrity check FAILED: {len(differences_found)} columns have unexpected changes")
        return False
    else:
        print(f"   ✅ Data integrity check PASSED: Only AGE column changed as expected")
        return True

def main():
    """Main execution function."""
    print("=" * 60)
    print("AGE ANONYMIZATION FOR HIPAA COMPLIANCE")
    print("=" * 60)
    print(f"Processing started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define input and output directories
    input_dir = "/lfs/skampere2/0/mahmedc/Comorbidities-Detection/datasets/full_data"
    output_dir = input_dir  # Same directory for output
    
    # Define file pairs (input, output)
    file_pairs = [
        ("train.csv", "train_age.csv"),
        ("val.csv", "val_age.csv"),
        ("test.csv", "test_age.csv")
    ]
    
    processed_files = []
    
    for input_file, output_file in file_pairs:
        input_path = os.path.join(input_dir, input_file)
        output_path = os.path.join(output_dir, output_file)
        
        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"❌ Input file not found: {input_path}")
            continue
        
        try:
            # Read the CSV file
            print(f"\n📖 Reading {input_file}...")
            df_original = pd.read_csv(input_path)
            print(f"   ✅ Loaded {len(df_original):,} records")
            
            # Anonymize age column
            df_anonymized = anonymize_age_column(df_original.copy(), input_file)
            
            # Verify data integrity
            integrity_check = verify_only_age_changed(df_original, df_anonymized, input_file)
            
            if integrity_check:
                # Save the anonymized file
                print(f"💾 Saving anonymized data to {output_file}...")
                df_anonymized.to_csv(output_path, index=False)
                print(f"   ✅ Saved {len(df_anonymized):,} records to {output_file}")
                
                processed_files.append((input_file, output_file, len(df_anonymized)))
            else:
                print(f"   ❌ Skipping save due to data integrity check failure")
                continue
            
        except Exception as e:
            print(f"❌ Error processing {input_file}: {e}")
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ AGE ANONYMIZATION COMPLETED!")
    print("=" * 60)
    print(f"📊 Summary of processed files:")
    for input_file, output_file, record_count in processed_files:
        print(f"   {input_file} → {output_file}: {record_count:,} records")
    
    print(f"\n📁 All output files saved in: {output_dir}")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
