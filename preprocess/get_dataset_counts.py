#!/usr/bin/env python3
"""
Get Dataset Counts After Processing
===================================

This script analyzes the biomarkers with HCC codes dataset to get the final counts
after all processing steps (filtering, etc.) that would go into train/val/test splits.

Author: AI Assistant
Date: 2025-01-27
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split

def get_final_counts():
    """Get the final counts after all processing steps."""
    print("🚀 GETTING DATASET COUNTS AFTER PROCESSING")
    print("=" * 60)
    
    # Load the dataset
    data_path = "/lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data/preprocess/biomarkers_with_hcc_codes_20250909_062523.csv"
    
    print("📊 Loading dataset...")
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Dataset loaded: {df.shape}")
        print(f"   Total records: {len(df):,}")
        print(f"   Total columns: {len(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Step 1: Process gender filtering (same as in modeling script)
    print("\n🔧 Step 1: Gender filtering...")
    if 'SEX' in df.columns:
        print(f"Original SEX distribution:")
        print(df['SEX'].value_counts())
        
        # Convert to lowercase and filter out Unknown values
        df['GENDER'] = df['SEX'].str.lower()
        initial_count = len(df)
        df = df[df['GENDER'].isin(['male', 'female'])].copy()
        filtered_count = len(df)
        
        print(f"✅ After gender filtering: {filtered_count:,} records")
        print(f"   Filtered out: {initial_count - filtered_count:,} Unknown records")
        print(f"Final GENDER distribution:")
        print(df['GENDER'].value_counts())
    else:
        print("❌ SEX column not found")
        return
    
    # Step 2: Get unique patients count
    print("\n👥 Step 2: Unique patients analysis...")
    if 'PAT-ID' in df.columns:
        unique_patients = df['PAT-ID'].unique()
        n_patients = len(unique_patients)
        
        print(f"📋 Total unique patients: {n_patients:,}")
        print(f"📋 Total records after filtering: {len(df):,}")
        print(f"📋 Average records per patient: {len(df)/n_patients:.2f}")
        
        # Show patient distribution
        patient_counts = df['PAT-ID'].value_counts()
        print(f"📊 Records per patient statistics:")
        print(f"   Min: {patient_counts.min()}")
        print(f"   Max: {patient_counts.max()}")
        print(f"   Mean: {patient_counts.mean():.2f}")
        print(f"   Median: {patient_counts.median():.1f}")
    else:
        print("❌ PAT-ID column not found")
        return
    
    # Step 3: Simulate train/val/test splits (without saving)
    print("\n📊 Step 3: Train/Val/Test split simulation...")
    
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
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    print(f"🎯 Total unique patients: {n_patients:,}")
    print(f"🎯 Total records after processing: {len(df):,}")
    print(f"🎯 Train/Val/Test record counts: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    print(f"🎯 Train/Val/Test patient counts: {len(train_patients):,}/{len(val_patients):,}/{len(test_patients):,}")
    
    return {
        'total_patients': n_patients,
        'total_records': len(df),
        'train_records': len(train_df),
        'val_records': len(val_df),
        'test_records': len(test_df),
        'train_patients': len(train_patients),
        'val_patients': len(val_patients),
        'test_patients': len(test_patients),
        'processed_df': df
    }

def analyze_pairwise_correlations(df):
    """Analyze pairwise correlations for all columns that go into train/val/test splits."""
    print("\n🔍 ANALYZING PAIRWISE CORRELATIONS")
    print("=" * 60)
    
    # Define the exact columns that are in the train/val/test CSV files
    final_columns = [
        'GENDER', 'AGE', 'MORTALITY', 'TYPE2DIABETES', 'ESSENTIALHTN', 'MI', 'HEARTFAILURE',
        'HCC12', 'HCC18', 'HCC19', 'HCC22', 'HCC48', 'HCC85', 'HCC96', 'HCC108', 'HCC111',
        'HCC_COMORBIDITY_SEVERITY', 'HCC_COMORBIDITY_SEVERITY_CATEGORY',
        'CALCIUMSCORING_ABDOMINALAGATSTON', 'CALCIUMSCORING_ABDOMINALAGATSTON_BINARY', 'CALCIUMSCORING_ABDOMINALAGATSTON_MULTICLASS',
        'KIDNEYVALUES_KIDNEYMEDIANHU', 'SPLEENVALUES_SPLEENVOLUME'
    ]
    
    # Create a processed dataframe that simulates the modeling script processing
    print("🔄 Simulating modeling script processing to create final columns...")
    processed_df = df.copy()
    
    # Process GENDER (already done in main function)
    if 'GENDER' not in processed_df.columns and 'SEX' in processed_df.columns:
        processed_df['GENDER'] = processed_df['SEX'].str.lower()
        processed_df = processed_df[processed_df['GENDER'].isin(['male', 'female'])].copy()
    
    # Process MORTALITY
    if 'DEATH_DATE' in processed_df.columns:
        processed_df['MORTALITY'] = processed_df['DEATH_DATE'].apply(lambda x: 'PRESENT' if pd.notna(x) else 'ABSENT')
    
    # Process diagnosis variables
    diagnosis_mapping = {
        'Type_2_Diabetes_DX_Code': 'TYPE2DIABETES',
        'essential_HTN_DX_Code': 'ESSENTIALHTN', 
        'MI_DX_Code': 'MI',
        'Heart_failure_DX_Code': 'HEARTFAILURE'
    }
    
    for original_col, new_col in diagnosis_mapping.items():
        if original_col in processed_df.columns:
            processed_df[new_col] = processed_df[original_col].apply(lambda x: 'PRESENT' if pd.notna(x) else 'ABSENT')
    
    # Process HCC codes
    hcc_cols = ['HCC_12', 'HCC_18', 'HCC_19', 'HCC_22', 'HCC_48', 'HCC_85', 'HCC_96', 'HCC_108', 'HCC_111']
    for col in hcc_cols:
        if col in processed_df.columns:
            new_col_name = col.replace('_', '')  # HCC_12 -> HCC12
            processed_df[new_col_name] = processed_df[col].apply(lambda x: 'PRESENT' if x > 0 else 'ABSENT')
    
    # Create HCC comorbidity severity
    hcc_binary_cols = ['HCC12', 'HCC18', 'HCC19', 'HCC22', 'HCC48', 'HCC85', 'HCC96', 'HCC108', 'HCC111']
    hcc_present_counts = []
    for _, row in processed_df.iterrows():
        present_count = sum(1 for col in hcc_binary_cols if row[col] == 'PRESENT')
        hcc_present_counts.append(present_count)
    
    processed_df['HCC_COMORBIDITY_SEVERITY'] = hcc_present_counts
    
    def severity_category(count):
        if count == 0:
            return 'NONE'
        elif 1 <= count <= 2:
            return 'LOW'
        elif 3 <= count <= 5:
            return 'MODERATE'
        else:
            return 'HIGH'
    
    processed_df['HCC_COMORBIDITY_SEVERITY_CATEGORY'] = processed_df['HCC_COMORBIDITY_SEVERITY'].apply(severity_category)
    
    # Process calcium scoring
    if 'CalciumScoring_AbdominalAgatston' in processed_df.columns:
        processed_df['CALCIUMSCORING_ABDOMINALAGATSTON'] = processed_df['CalciumScoring_AbdominalAgatston']
        
        # Binary variable (>1000 = PRESENT)
        processed_df['CALCIUMSCORING_ABDOMINALAGATSTON_BINARY'] = processed_df['CALCIUMSCORING_ABDOMINALAGATSTON'].apply(
            lambda x: 'PRESENT' if x > 1000 else 'ABSENT'
        )
        
        # Multiclass variable
        def calcium_multiclass(x):
            if x == 0:
                return 'ABSENT'
            elif 0 < x <= 1000:
                return 'LOW'
            elif 1000 < x <= 3000:
                return 'MEDIUM'
            else:
                return 'HIGH'
        
        processed_df['CALCIUMSCORING_ABDOMINALAGATSTON_MULTICLASS'] = processed_df['CALCIUMSCORING_ABDOMINALAGATSTON'].apply(calcium_multiclass)
    
    # Process biomarker variables
    if 'KidneyValues_KidneyMedianHU' in processed_df.columns:
        processed_df['KIDNEYVALUES_KIDNEYMEDIANHU'] = processed_df['KidneyValues_KidneyMedianHU']
    
    if 'SpleenValues_SpleenVolume' in processed_df.columns:
        processed_df['SPLEENVALUES_SPLEENVOLUME'] = processed_df['SpleenValues_SpleenVolume']
    
    # Get available columns from the processed dataframe
    available_columns = [col for col in final_columns if col in processed_df.columns]
    missing_columns = [col for col in final_columns if col not in processed_df.columns]
    
    print(f"📋 Available columns for correlation analysis: {len(available_columns)}")
    if missing_columns:
        print(f"⚠️  Missing columns: {missing_columns}")
    
    # Use the processed dataframe for analysis
    analysis_columns = available_columns
    
    # Categorize columns by type
    numeric_columns = []
    binary_columns = []
    categorical_columns = []
    
    for col in analysis_columns:
        if col in processed_df.columns:
            # Check if column has enough non-null values
            non_null_pct = processed_df[col].notna().sum() / len(processed_df) * 100
            if non_null_pct >= 50:  # Only include columns with at least 50% non-null values
                if processed_df[col].dtype in ['int64', 'float64']:
                    numeric_columns.append(col)
                else:
                    unique_vals = processed_df[col].nunique()
                    if unique_vals == 2:
                        binary_columns.append(col)
                    elif 2 < unique_vals <= 10:  # Categorical with few categories
                        categorical_columns.append(col)
    
    print(f"📊 Numeric columns: {len(numeric_columns)}")
    print(f"📊 Binary columns: {len(binary_columns)}")
    print(f"📊 Categorical columns: {len(categorical_columns)}")
    
    if len(binary_columns) > 0:
        print(f"   Binary columns: {binary_columns}")
    
    if len(categorical_columns) > 0:
        print(f"   Categorical columns: {categorical_columns}")
    
    # Combine all columns for analysis
    all_analysis_columns = numeric_columns + binary_columns + categorical_columns
    
    if len(all_analysis_columns) < 2:
        print("❌ Not enough columns for correlation analysis")
        return
    
    # Prepare data for correlation analysis
    print("🔄 Preparing data for correlation analysis...")
    
    # Create a copy of the processed dataframe for analysis
    analysis_df = processed_df[all_analysis_columns].copy()
    
    # Convert binary and categorical variables to numeric for correlation analysis
    for col in binary_columns + categorical_columns:
        if col in analysis_df.columns:
            # For binary variables, convert to 0/1
            if col in binary_columns:
                unique_vals = sorted(analysis_df[col].dropna().unique())
                if len(unique_vals) == 2:
                    # Create binary encoding
                    val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
                    analysis_df[col] = analysis_df[col].map(val_map)
            else:
                # For categorical variables, use label encoding
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                non_null_mask = analysis_df[col].notna()
                analysis_df.loc[non_null_mask, col] = le.fit_transform(analysis_df.loc[non_null_mask, col])
    
    # Calculate correlation matrix
    print("🔄 Calculating correlation matrix...")
    corr_matrix = analysis_df.corr()
    
    # Create output directory
    output_dir = "/lfs/turing1/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection/preprocess/eda_results_biomarkers_hcc"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save correlation matrix as CSV
    corr_csv_path = os.path.join(output_dir, "final_modeling_columns_correlations.csv")
    corr_matrix.to_csv(corr_csv_path)
    print(f"✅ Final modeling columns correlation matrix saved: {corr_csv_path}")
    
    # Find high correlations
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.7 and not np.isnan(corr_val):
                high_corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    corr_val
                ))
    
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"\n📊 High correlations (|r| > 0.7): {len(high_corr_pairs)} pairs")
    for i, (col1, col2, corr_val) in enumerate(high_corr_pairs[:10]):
        print(f"   {i+1:2d}. {col1[:30]} <-> {col2[:30]}: {corr_val:.3f}")
    
    # Save high correlations as CSV
    if high_corr_pairs:
        high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Variable_1', 'Variable_2', 'Correlation'])
        high_corr_csv_path = os.path.join(output_dir, "final_modeling_columns_high_correlations.csv")
        high_corr_df.to_csv(high_corr_csv_path, index=False)
        print(f"✅ High correlations saved: {high_corr_csv_path}")
    
    # Separate analysis for binary variables
    if len(binary_columns) > 1:
        print(f"\n🔍 BINARY VARIABLE CORRELATION ANALYSIS")
        print("=" * 50)
        
        # Create binary-only correlation matrix
        binary_analysis_df = analysis_df[binary_columns].copy()
        binary_corr_matrix = binary_analysis_df.corr()
        
        # Save binary correlation matrix
        binary_corr_csv_path = os.path.join(output_dir, "final_modeling_binary_correlations.csv")
        binary_corr_matrix.to_csv(binary_corr_csv_path)
        print(f"✅ Binary variables correlation matrix saved: {binary_corr_csv_path}")
        
        # Find high binary correlations
        binary_high_corr_pairs = []
        for i in range(len(binary_corr_matrix.columns)):
            for j in range(i+1, len(binary_corr_matrix.columns)):
                corr_val = binary_corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.3 and not np.isnan(corr_val):  # Lower threshold for binary
                    binary_high_corr_pairs.append((
                        binary_corr_matrix.columns[i], 
                        binary_corr_matrix.columns[j], 
                        corr_val
                    ))
        
        binary_high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print(f"📊 Binary variable correlations (|r| > 0.3): {len(binary_high_corr_pairs)} pairs")
        for i, (col1, col2, corr_val) in enumerate(binary_high_corr_pairs[:10]):
            print(f"   {i+1:2d}. {col1[:30]} <-> {col2[:30]}: {corr_val:.3f}")
        
        # Save binary high correlations
        if binary_high_corr_pairs:
            binary_high_corr_df = pd.DataFrame(binary_high_corr_pairs, columns=['Variable_1', 'Variable_2', 'Correlation'])
            binary_high_corr_csv_path = os.path.join(output_dir, "final_modeling_binary_high_correlations.csv")
            binary_high_corr_df.to_csv(binary_high_corr_csv_path, index=False)
            print(f"✅ Binary high correlations saved: {binary_high_corr_csv_path}")
        
        # Create binary correlation heatmap
        if len(binary_columns) <= 50:  # Only create heatmap if not too many variables
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(binary_corr_matrix, dtype=bool))
            
            sns.heatmap(binary_corr_matrix, 
                        mask=mask,
                        annot=True,  # Show correlation values for binary
                        cmap='coolwarm', 
                        center=0,
                        square=True,
                        cbar_kws={"shrink": .8},
                        fmt='.2f')
            
            plt.title(f'Final Modeling Binary Variables Correlations\n({len(binary_columns)} variables)', fontsize=14, pad=20)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save binary heatmap
            binary_heatmap_path = os.path.join(output_dir, "final_modeling_binary_correlations_heatmap.png")
            plt.savefig(binary_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Binary correlation heatmap saved: {binary_heatmap_path}")
        else:
            print(f"⚠️  Too many binary variables ({len(binary_columns)}) for heatmap visualization")
    
    # Create correlation heatmap
    print("🎨 Creating final modeling columns correlation heatmap...")
    
    # For large correlation matrices, sample a subset for visualization
    if len(all_analysis_columns) > 30:
        # Select top 30 columns with most data
        column_data_counts = analysis_df.notna().sum()
        top_columns = column_data_counts.nlargest(30).index.tolist()
        sample_corr_matrix = corr_matrix.loc[top_columns, top_columns]
        print(f"   Showing top 30 columns with most data for heatmap")
    else:
        sample_corr_matrix = corr_matrix
    
    # Create the heatmap
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(sample_corr_matrix, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(sample_corr_matrix, 
                mask=mask,
                annot=False,  # Don't show correlation values (too cluttered)
                cmap='coolwarm', 
                center=0,
                square=True,
                cbar_kws={"shrink": .8})
    
    plt.title(f'Final Modeling Columns Correlations Heatmap\n({len(sample_corr_matrix)} variables)', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, "final_modeling_columns_correlations_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Final modeling columns correlation heatmap saved: {heatmap_path}")
    
    # Summary statistics
    print(f"\n📊 Correlation Analysis Summary:")
    print(f"   Total variables analyzed: {len(all_analysis_columns)}")
    print(f"   - Numeric variables: {len(numeric_columns)}")
    print(f"   - Binary variables: {len(binary_columns)}")
    print(f"   - Categorical variables: {len(categorical_columns)}")
    print(f"   Total variable pairs analyzed: {len(all_analysis_columns) * (len(all_analysis_columns) - 1) // 2}")
    print(f"   High correlations (|r| > 0.7): {len(high_corr_pairs)}")
    print(f"   High correlations (|r| > 0.5): {sum(1 for i in range(len(corr_matrix.columns)) for j in range(i+1, len(corr_matrix.columns)) if abs(corr_matrix.iloc[i, j]) > 0.5 and not np.isnan(corr_matrix.iloc[i, j]))}")
    print(f"   High correlations (|r| > 0.3): {sum(1 for i in range(len(corr_matrix.columns)) for j in range(i+1, len(corr_matrix.columns)) if abs(corr_matrix.iloc[i, j]) > 0.3 and not np.isnan(corr_matrix.iloc[i, j]))}")
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlations': high_corr_pairs,
        'numeric_columns': numeric_columns,
        'binary_columns': binary_columns,
        'categorical_columns': categorical_columns,
        'all_analysis_columns': all_analysis_columns
    }

if __name__ == "__main__":
    counts = get_final_counts()
    
    # Run correlation analysis
    if 'processed_df' in counts:
        correlation_results = analyze_pairwise_correlations(counts['processed_df'])
        print(f"\n🎯 Correlation analysis completed!")
        print(f"   Results saved to: /lfs/turing1/0/mahmedc/Comorbidities-Detection/CT-Disease-Detection/preprocess/eda_results_biomarkers_hcc/")
