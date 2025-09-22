#!/usr/bin/env python3
"""
HIPAA Compliance and Data Privacy Script
========================================

This script implements HIPAA compliance measures for the comorbidities detection dataset.

Author: AI Assistant
Date: 2025-01-27
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import hashlib
import uuid

def load_input_data():
    """Load the input dataset used by 07_modeling_dataset_with_splits.py"""
    print("🚀 HIPAA COMPLIANCE AND DATA PRIVACY PROCESSING")
    print("=" * 60)
    
    # Same input file as used in 07_modeling_dataset_with_splits.py
    data_path = "../../datasets/full_data/biomarkers_with_hcc_codes_20250909_062523.csv"
    
    print("📊 Loading input dataset...")
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"   Original records: {len(df):,}")
        print(f"   Original columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    return df

def remove_fully_missing_columns(df):
    """Remove columns that are 100% missing (all NaN/null values)."""
    print("\n🧹 Removing columns with 100% missing values...")
    
    initial_columns = len(df.columns)
    
    # Check for columns with 100% missing values
    missing_percentages = df.isnull().sum() / len(df) * 100
    fully_missing_cols = missing_percentages[missing_percentages == 100.0].index.tolist()
    
    if fully_missing_cols:
        print(f"📋 Found {len(fully_missing_cols)} columns with 100% missing values:")
        for col in fully_missing_cols:
            print(f"   - {col}")
        
        # Remove fully missing columns
        df = df.drop(columns=fully_missing_cols)
        print(f"✅ Removed {len(fully_missing_cols)} fully missing columns")
    else:
        print("✅ No columns with 100% missing values found")
    
    final_columns = len(df.columns)
    print(f"📊 Columns: {initial_columns} → {final_columns} (-{initial_columns - final_columns})")
    
    return df

def remove_unknown_gender_records(df):
    """Remove records where SEX/GENDER is unknown."""
    print("\n👤 Removing records with unknown gender...")
    
    initial_records = len(df)
    
    # Check if SEX column exists
    if 'SEX' in df.columns:
        print(f"📋 Original SEX distribution:")
        sex_counts = df['SEX'].value_counts(dropna=False)
        print(sex_counts)
        
        # Filter out unknown/null gender values
        # Keep only 'male' and 'female' (case-insensitive)
        df = df[df['SEX'].str.lower().isin(['male', 'female'])].copy()
        
        final_records = len(df)
        removed_records = initial_records - final_records
        
        print(f"✅ Removed {removed_records:,} records with unknown gender")
        print(f"📊 Records: {initial_records:,} → {final_records:,} (-{removed_records:,})")
        
        # Show final gender distribution
        print(f"📋 Final gender distribution:")
        final_sex_counts = df['SEX'].value_counts()
        print(final_sex_counts)
        
    else:
        print("❌ SEX column not found in dataset")
        return None
    
    return df

def confirm_record_count(df, expected_count=23506):
    """Confirm the resulting record count matches expected value."""
    print(f"\n✅ CONFIRMING RECORD COUNT")
    print("=" * 30)
    
    actual_count = len(df)
    print(f"📊 Expected records: {expected_count:,}")
    print(f"📊 Actual records: {actual_count:,}")
    
    if actual_count == expected_count:
        print(f"✅ SUCCESS: Record count matches expected value ({expected_count:,})")
        return True
    else:
        print(f"❌ MISMATCH: Expected {expected_count:,} but got {actual_count:,}")
        print(f"   Difference: {actual_count - expected_count:,}")
        return False

def remove_hipaa_sensitive_columns(df):
    """Remove columns containing patient names and geographic subdivisions smaller than a state."""
    print("\n🔒 Removing HIPAA-sensitive columns...")
    
    initial_columns = len(df.columns)
    
    # Define patterns for HIPAA-sensitive columns
    patient_name_patterns = [
        'name', 'first_name', 'last_name', 'patient_name', 'full_name',
        'given_name', 'family_name', 'surname', 'forename'
    ]
    
    geographic_patterns = [
        'address', 'street', 'city', 'county', 'precinct', 'zip', 'postal',
        'geocode', 'latitude', 'longitude', 'lat', 'lng', 'coord',
        'location', 'residence', 'home_address', 'mailing_address'
    ]
    
    # Additional patterns for medical record identifiers that might contain names
    medical_id_patterns = [
        'mrn', 'medical_record', 'patient_id', 'subject_id', 'patient_name'
    ]
    
    # Combine all patterns
    all_sensitive_patterns = patient_name_patterns + geographic_patterns + medical_id_patterns
    
    # Find columns to remove
    columns_to_remove = []
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check if column name contains any sensitive pattern
        for pattern in all_sensitive_patterns:
            if pattern in col_lower:
                columns_to_remove.append(col)
                break
        
        # Additional specific checks for common problematic columns
        if any(keyword in col_lower for keyword in ['_name', '_address', '_location']):
            if col not in columns_to_remove:
                columns_to_remove.append(col)
    
    # Remove the identified columns
    if columns_to_remove:
        print(f"📋 Found {len(columns_to_remove)} HIPAA-sensitive columns to remove:")
        for col in sorted(columns_to_remove):
            print(f"   - {col}")
        
        # Remove columns
        df = df.drop(columns=columns_to_remove)
        print(f"✅ Removed {len(columns_to_remove)} HIPAA-sensitive columns")
    else:
        print("✅ No HIPAA-sensitive columns found")
    
    final_columns = len(df.columns)
    print(f"📊 Columns: {initial_columns} → {final_columns} (-{initial_columns - final_columns})")
    
    return df, columns_to_remove

def remove_specific_hipaa_columns(df, columns_to_remove):
    """Remove specific HIPAA-sensitive columns (safe with only 49 duplicates in ACC_NUM-SESSION-ID)."""
    print(f"\n🗑️  Removing specific HIPAA-sensitive columns...")
    
    initial_columns = len(df.columns)
    
    # Filter to only remove columns that actually exist
    existing_columns = [col for col in columns_to_remove if col in df.columns]
    missing_columns = [col for col in columns_to_remove if col not in df.columns]
    
    if missing_columns:
        print(f"⚠️  Columns not found in dataset: {missing_columns}")
    
    if existing_columns:
        print(f"📋 Removing {len(existing_columns)} specific HIPAA-sensitive columns:")
        for col in existing_columns:
            print(f"   - {col}")
        
        # Remove the columns
        df = df.drop(columns=existing_columns)
        print(f"✅ Removed {len(existing_columns)} specific HIPAA-sensitive columns")
    else:
        print("✅ No specific HIPAA-sensitive columns found to remove")
    
    final_columns = len(df.columns)
    print(f"📊 Columns: {initial_columns} → {final_columns} (-{initial_columns - final_columns})")
    
    return df, existing_columns

def anonymize_dates_and_ages(df):
    """Anonymize dates (keep only year) and ages (convert >=90 to '90+')."""
    print("\n📅 Anonymizing dates and ages for HIPAA compliance...")
    
    # Define date columns to anonymize (keep only year)
    date_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['date', 'dt']) and col != 'AGE':
            date_columns.append(col)
    
    print(f"📋 Found {len(date_columns)} date columns to anonymize")
    
    # Process each date column
    anonymized_date_columns = []
    for col in date_columns:
        if col in df.columns:
            non_null_count = df[col].notna().sum()
            if non_null_count > 0:
                try:
                    # Convert to datetime first
                    df[f'{col}_temp'] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')
                    
                    # Extract year only
                    df[col] = df[f'{col}_temp'].dt.year
                    
                    # Remove temporary column
                    df = df.drop(columns=[f'{col}_temp'])
                    
                    anonymized_date_columns.append(col)
                    print(f"✅ {col}: converted to year-only format ({non_null_count:,} values)")
                    
                except Exception as e:
                    print(f"⚠️  {col}: failed to anonymize ({e})")
            else:
                print(f"⚠️  {col}: no values to process")
    
    # Process AGE column - convert ages >= 90 to "90+"
    print(f"\n👤 Processing AGE column...")
    if 'AGE' in df.columns:
        initial_age_count = len(df)
        ages_90_plus = (df['AGE'] >= 90).sum()
        
        # Convert ages >= 90 to "90+"
        df['AGE'] = df['AGE'].apply(lambda x: "90+" if x >= 90 else x)
        
        print(f"✅ AGE: converted {ages_90_plus:,} ages >= 90 to '90+'")
        print(f"   Total records processed: {initial_age_count:,}")
        
        # Show new age distribution
        age_distribution = df['AGE'].value_counts().sort_index()
        print(f"📊 Age distribution after anonymization:")
        for age, count in age_distribution.items():
            if str(age).startswith('90') or isinstance(age, str):  # Show 90+ category
                print(f"   Age {age}: {count:,} records")
    else:
        print("⚠️  AGE column not found")
    
    print(f"✅ Date/age anonymization completed:")
    print(f"   - {len(anonymized_date_columns)} date columns anonymized to year-only")
    print(f"   - AGE column: ages >= 90 converted to '90+'")
    
    return df, anonymized_date_columns

def main():
    """Main execution function."""
    print("=" * 80)
    print("HIPAA COMPLIANCE: DATA CLEANING AND PRIVACY FILTERING")
    print("=" * 80)
    print(f"Processing started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Load input data
    df = load_input_data()
    if df is None:
        return
    
    # 2. Remove fully missing columns
    df = remove_fully_missing_columns(df)
    
    # 3. Remove unknown gender records
    df = remove_unknown_gender_records(df)
    if df is None:
        return
    
    # 4. Remove HIPAA-sensitive columns
    df, removed_columns = remove_hipaa_sensitive_columns(df)
    
    # 5. Remove specific additional HIPAA-sensitive columns (safe with only 49 duplicates)
    specific_columns_to_remove = [
        'subject.label',
        'StudyInfo_MRN', 
        'PAT_ID_MASKED',
        'ACC_NUM_EXTRACTED',
        'ACC_NUM'
    ]
    df, additional_removed = remove_specific_hipaa_columns(df, specific_columns_to_remove)
    print(f"✅ Additional HIPAA columns removed: {len(additional_removed)} columns")
    
    # 6. Anonymize dates and ages
    df, anonymized_dates = anonymize_dates_and_ages(df)
    
    # 7. Confirm final record count
    success = confirm_record_count(df, expected_count=23506)
    
    print("\n" + "=" * 80)
    if success:
        print("✅ HIPAA COMPLIANCE PROCESSING COMPLETED SUCCESSFULLY!")
        print(f"📊 Final dataset: {len(df):,} records × {len(df.columns)} columns")
        print(f"🎯 Ready for further processing!")
    else:
        print("⚠️  HIPAA COMPLIANCE PROCESSING COMPLETED WITH ISSUES")
        print("   Please review the record count mismatch")
    
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
