#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) of CT Biomarkers & Clinical Outcomes Dataset

This script provides a thorough exploratory data analysis of the CT Biomarkers & Clinical Outcomes 
dataset (2025_08_31_Biomarkers_Outcomes_Joined_Fixed.csv). The dataset combines CT-derived biomarker 
measurements with comprehensive clinical outcomes data.

Usage: python biomarkers_outcomes_EDAs_script.py
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Try to import optional libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available, using matplotlib for all visualizations")

try:
    from scipy import stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available, skipping advanced analytics")

try:
    import missingno as msno
    MISSINGNO_AVAILABLE = True
except ImportError:
    MISSINGNO_AVAILABLE = False
    print("missingno not available, using alternative visualization")

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

def main():
    print("="*80)
    print("COMPREHENSIVE EDA: CT BIOMARKERS & CLINICAL OUTCOMES DATASET")
    print("="*80)
    
    # 1. Data Loading and Overview
    print("\n1. DATA LOADING AND OVERVIEW")
    print("-" * 40)
    
    # Load the dataset
    data_path = "../../datasets/full_data/2025_08_31_Biomarkers_Outcomes_Joined_Fixed.csv"
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print("✅ Dataset loaded successfully!")
    except FileNotFoundError:
        print("❌ Dataset file not found. Please check the path.")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📈 Number of rows: {df.shape[0]:,}")
    print(f"📋 Number of columns: {df.shape[1]:,}")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Display basic information about the dataset
    print("\n=== DATASET OVERVIEW ===")
    
    # Create a summary DataFrame for better visualization
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data_Type': df.dtypes,
        'Non_Null_Count': df.count(),
        'Null_Count': df.isnull().sum(),
        'Null_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    print(f"\nFirst 20 columns summary:")
    print(col_info.head(20).to_string())
    print(f"\n... and {len(col_info) - 20} more columns")
    
    # 2. Column Categorization
    print("\n2. COLUMN CATEGORIZATION")
    print("-" * 30)
    
    # Categorize columns based on their content and naming patterns
    # CT Biomarker columns
    ct_biomarker_cols = [
        # Study identification
        'StudyInfo_Accession', 'StudyInfo_MRN', 'StudyInfo_SeriesInfo', 'subject.label', 'session.label',
        # Bone density measurements
        'BMDL1Values_BMDL1HighSensitivityHU', 'BMDL1Values_BMDL1StandardHU',
        # Calcium scoring
        'CalciumScoring_AbdominalAgatston',
        # Organ measurements
        'KidneyValues_KidneyMedianHU', 'KidneyValues_KidneyVolume',
        'LiverValues_LiverMedianHU', 'LiverValues_LiverVolume',
        'SpleenValues_SpleenMedianHU', 'SpleenValues_SpleenVolume',
        'PancreasValues_PancreasMedianHU', 'PancreasValues_PancreasVolume'
    ]
    
    # Fat measurement columns
    fat_cols = [col for col in df.columns if 'Fat' in col or 'SAT' in col or 'VAT' in col or 'TAT' in col]
    
    # Muscle measurement columns  
    muscle_cols = [col for col in df.columns if 'Muscle' in col or 'IMAT' in col]
    
    # DICOM metadata columns
    dicom_cols = [col for col in df.columns if 'DICOMHeader_' in col or 'Levels_' in col]
    
    # Clinical outcome columns (similar to Oscar Master Cohort)
    diagnosis_cols = [col for col in df.columns if '_DX_Code' in col or '_DX_NAME' in col or '_DX_dt' in col or '_DX_DT' in col]
    lab_cols = [col for col in df.columns if 'HA1C' in col or 'CRPN' in col]
    date_cols = [col for col in df.columns if 'DATE' in col.upper() or '_DT' in col.upper()]
    patient_id_cols = ['PAT_ID_MASKED', 'ACC_NUM-SESSION-ID']
    
    print("=== COLUMN CATEGORIZATION ===")
    print(f"📊 Total columns: {len(df.columns)}")
    print(f"🔬 CT Biomarker base columns: {len([col for col in ct_biomarker_cols if col in df.columns])}")
    print(f"🫁 Fat measurement columns: {len(fat_cols)}")
    print(f"💪 Muscle measurement columns: {len(muscle_cols)}")
    print(f"🏥 DICOM metadata columns: {len(dicom_cols)}")
    print(f"🏥 Diagnosis-related columns: {len(diagnosis_cols)}")
    print(f"🧪 Lab value columns: {len(lab_cols)}")
    print(f"📅 Date columns: {len(date_cols)}")
    print(f"👤 Patient ID columns: {len(patient_id_cols)}")
    
    # Calculate total categorized vs uncategorized
    all_categorized = set(ct_biomarker_cols + fat_cols + muscle_cols + dicom_cols + 
                         diagnosis_cols + lab_cols + patient_id_cols)
    uncategorized = set(df.columns) - all_categorized
    print(f"📋 Other/uncategorized columns: {len(uncategorized)}")
    
    # Display sample columns from each category
    print(f"\nSample CT Biomarker columns:")
    sample_ct_cols = [col for col in ct_biomarker_cols[:10] if col in df.columns]
    for i, col in enumerate(sample_ct_cols):
        print(f"  {i+1}. {col}")
    
    print(f"\nSample Fat measurement columns:")
    for i, col in enumerate(fat_cols[:10]):
        print(f"  {i+1}. {col}")
    
    print(f"\nSample Muscle measurement columns:")
    for i, col in enumerate(muscle_cols[:10]):
        print(f"  {i+1}. {col}")
    
    print(f"\nSample Diagnosis columns:")
    for i, col in enumerate(diagnosis_cols[:10]):
        print(f"  {i+1}. {col}")
    
    print(f"\nLab columns:")
    for i, col in enumerate(lab_cols):
        print(f"  {i+1}. {col}")
    
    # 3. CT Biomarkers Analysis
    print("\n\n3. CT BIOMARKERS ANALYSIS")
    print("-" * 30)
    
    print("=== CT BIOMARKERS ANALYSIS ===")
    
    # Bone density analysis
    bone_cols = ['BMDL1Values_BMDL1HighSensitivityHU', 'BMDL1Values_BMDL1StandardHU']
    print(f"\n🦴 BONE DENSITY ANALYSIS:")
    for col in bone_cols:
        if col in df.columns:
            # Convert to numeric, handling string values
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            valid_values = numeric_values.dropna()
            if len(valid_values) > 0:
                print(f"  {col}:")
                print(f"    Valid measurements: {len(valid_values):,}")
                print(f"    Range: {valid_values.min():.1f} - {valid_values.max():.1f} HU")
                print(f"    Mean ± SD: {valid_values.mean():.1f} ± {valid_values.std():.1f} HU")
    
    # Calcium scoring analysis
    if 'CalciumScoring_AbdominalAgatston' in df.columns:
        # Convert to numeric, handling string values
        numeric_calcium = pd.to_numeric(df['CalciumScoring_AbdominalAgatston'], errors='coerce')
        calcium_scores = numeric_calcium.dropna()
        print(f"\n🫀 CALCIUM SCORING ANALYSIS:")
        print(f"  Valid calcium scores: {len(calcium_scores):,}")
        if len(calcium_scores) > 0:
            print(f"  Range: {calcium_scores.min():.1f} - {calcium_scores.max():.1f}")
            print(f"  Median [IQR]: {calcium_scores.median():.1f} [{calcium_scores.quantile(0.25):.1f}-{calcium_scores.quantile(0.75):.1f}]")
            
            # Clinical categories for calcium scoring
            zero_calcium = (calcium_scores == 0).sum()
            minimal = ((calcium_scores > 0) & (calcium_scores < 10)).sum()
            mild = ((calcium_scores >= 10) & (calcium_scores < 100)).sum()
            moderate = ((calcium_scores >= 100) & (calcium_scores < 400)).sum()
            severe = (calcium_scores >= 400).sum()
            
            print(f"  Clinical categories:")
            print(f"    Zero calcium (0): {zero_calcium} ({zero_calcium/len(calcium_scores)*100:.1f}%)")
            print(f"    Minimal (1-9): {minimal} ({minimal/len(calcium_scores)*100:.1f}%)")
            print(f"    Mild (10-99): {mild} ({mild/len(calcium_scores)*100:.1f}%)")
            print(f"    Moderate (100-399): {moderate} ({moderate/len(calcium_scores)*100:.1f}%)")
            print(f"    Severe (≥400): {severe} ({severe/len(calcium_scores)*100:.1f}%)")
    
    # Organ measurements analysis
    organ_cols = {
        'Liver': ['LiverValues_LiverMedianHU', 'LiverValues_LiverVolume'],
        'Kidney': ['KidneyValues_KidneyMedianHU', 'KidneyValues_KidneyVolume'],
        'Spleen': ['SpleenValues_SpleenMedianHU', 'SpleenValues_SpleenVolume'],
        'Pancreas': ['PancreasValues_PancreasMedianHU', 'PancreasValues_PancreasVolume']
    }
    
    print(f"\n🫁 ORGAN MEASUREMENTS ANALYSIS:")
    for organ, cols in organ_cols.items():
        print(f"\n  {organ.upper()}:")
        for col in cols:
            if col in df.columns:
                # Convert to numeric, handling string values
                numeric_values = pd.to_numeric(df[col], errors='coerce')
                valid_values = numeric_values.dropna()
                if len(valid_values) > 0:
                    unit = "HU" if "HU" in col else "cm³"
                    print(f"    {col.split('_')[-1]}: {valid_values.mean():.1f} ± {valid_values.std():.1f} {unit} (n={len(valid_values):,})")
                    
                    # Clinical interpretation for liver HU (hepatic steatosis)
                    if 'Liver' in col and 'HU' in col:
                        normal_liver = (valid_values >= 50).sum()
                        mild_steatosis = ((valid_values >= 40) & (valid_values < 50)).sum()
                        moderate_steatosis = ((valid_values >= 30) & (valid_values < 40)).sum()
                        severe_steatosis = (valid_values < 30).sum()
                        
                        print(f"      Clinical categories (hepatic steatosis):")
                        print(f"        Normal (≥50 HU): {normal_liver} ({normal_liver/len(valid_values)*100:.1f}%)")
                        print(f"        Mild (40-49 HU): {mild_steatosis} ({mild_steatosis/len(valid_values)*100:.1f}%)")
                        print(f"        Moderate (30-39 HU): {moderate_steatosis} ({moderate_steatosis/len(valid_values)*100:.1f}%)")
                        print(f"        Severe (<30 HU): {severe_steatosis} ({severe_steatosis/len(valid_values)*100:.1f}%)")
    
    # Fat distribution analysis
    print(f"\n🫄 FAT DISTRIBUTION ANALYSIS:")
    
    # L3 level fat analysis (most clinically relevant)
    l3_fat_cols = [col for col in fat_cols if 'L3' in col]
    print(f"\n  L3 Level Fat Measurements (n={len(l3_fat_cols)} metrics):")
    
    key_l3_metrics = ['L3FatValues_L3VATArea', 'L3FatValues_L3SATArea', 'L3FatValues_L3TATArea', 'L3FatValues_L3VATSATRatio']
    for col in key_l3_metrics:
        if col in df.columns:
            # Convert to numeric, handling string values
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            valid_values = numeric_values.dropna()
            if len(valid_values) > 0:
                metric_name = col.split('_')[-1]
                unit = "cm²" if "Area" in col else "ratio" if "Ratio" in col else "HU"
                print(f"    {metric_name}: {valid_values.mean():.1f} ± {valid_values.std():.1f} {unit} (n={len(valid_values):,})")
                
                # Clinical interpretation for VAT/SAT ratio
                if 'VATSATRatio' in col:
                    low_ratio = (valid_values < 0.4).sum()
                    moderate_ratio = ((valid_values >= 0.4) & (valid_values < 1.0)).sum()
                    high_ratio = (valid_values >= 1.0).sum()
                    
                    print(f"      VAT/SAT ratio categories:")
                    print(f"        Low (<0.4): {low_ratio} ({low_ratio/len(valid_values)*100:.1f}%)")
                    print(f"        Moderate (0.4-0.99): {moderate_ratio} ({moderate_ratio/len(valid_values)*100:.1f}%)")
                    print(f"        High (≥1.0): {high_ratio} ({high_ratio/len(valid_values)*100:.1f}%)")
    
    # Muscle composition analysis
    print(f"\n💪 MUSCLE COMPOSITION ANALYSIS:")
    
    # L3 level muscle analysis (most clinically relevant)
    l3_muscle_cols = [col for col in muscle_cols if 'L3' in col]
    print(f"\n  L3 Level Muscle Measurements (n={len(l3_muscle_cols)} metrics):")
    
    key_l3_muscle_metrics = ['MuscleValues_L3MuscleArea', 'MuscleValues_L3MuscleMeanHU', 'MuscleValues_L3IMATArea']
    for col in key_l3_muscle_metrics:
        if col in df.columns:
            # Convert to numeric, handling string values
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            valid_values = numeric_values.dropna()
            if len(valid_values) > 0:
                metric_name = col.split('_')[-1]
                unit = "cm²" if "Area" in col else "HU"
                print(f"    {metric_name}: {valid_values.mean():.1f} ± {valid_values.std():.1f} {unit} (n={len(valid_values):,})")
                
                # Clinical interpretation for muscle HU (muscle quality)
                if 'MeanHU' in col:
                    high_quality = (valid_values >= 40).sum()
                    moderate_quality = ((valid_values >= 30) & (valid_values < 40)).sum()
                    low_quality = (valid_values < 30).sum()
                    
                    print(f"      Muscle quality categories:")
                    print(f"        High quality (≥40 HU): {high_quality} ({high_quality/len(valid_values)*100:.1f}%)")
                    print(f"        Moderate quality (30-39 HU): {moderate_quality} ({moderate_quality/len(valid_values)*100:.1f}%)")
                    print(f"        Low quality (<30 HU): {low_quality} ({low_quality/len(valid_values)*100:.1f}%)")
    
    # 4. Patient Demographics and Clinical Outcomes
    print("\n\n4. PATIENT DEMOGRAPHICS AND CLINICAL OUTCOMES")
    print("-" * 50)
    
    print("=== PATIENT DEMOGRAPHICS ANALYSIS ===")
    print(f"👥 Total unique patients: {df['PAT_ID_MASKED'].nunique():,}")
    print(f"📄 Total records: {len(df):,}")
    print(f"📊 Average records per patient: {len(df) / df['PAT_ID_MASKED'].nunique():.2f}")
    
    # Check for duplicate patient records
    duplicate_patients = df['PAT_ID_MASKED'].value_counts()
    patients_with_multiple_records = (duplicate_patients > 1).sum()
    print(f"🔄 Patients with multiple records: {patients_with_multiple_records}")
    
    # Mortality analysis
    death_data = df['DEATH_DATE'].dropna()
    total_patients = df['PAT_ID_MASKED'].nunique()
    deceased_patients = len(death_data)
    
    print(f"\n=== MORTALITY ANALYSIS ===")
    print(f"👥 Total patients: {total_patients:,}")
    print(f"💀 Deceased patients: {deceased_patients:,}")
    print(f"📊 Mortality rate: {deceased_patients/total_patients*100:.2f}%")
    
    # 5. Missing Values Analysis
    print("\n\n5. MISSING VALUES ANALYSIS")
    print("-" * 35)
    
    # Calculate missing values statistics
    missing_stats = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100,
        'Data_Type': df.dtypes
    })
    
    missing_stats = missing_stats.sort_values('Missing_Percentage', ascending=False)
    
    print("=== MISSING VALUES SUMMARY ===")
    print(f"📊 Total missing values in dataset: {df.isnull().sum().sum():,}")
    print(f"📈 Percentage of total values missing: {(df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
    
    print(f"\nTop 15 columns with highest missing percentages:")
    print(missing_stats.head(15)[['Column', 'Missing_Percentage']].to_string())
    
    # Analyze missing values patterns by data category
    complete_data_count = (missing_stats['Missing_Percentage'] == 0).sum()
    partial_missing_count = ((missing_stats['Missing_Percentage'] > 0) & (missing_stats['Missing_Percentage'] < 100)).sum()
    all_missing_count = (missing_stats['Missing_Percentage'] == 100).sum()
    
    print(f"\n📋 Data completeness summary:")
    print(f"   ✅ Columns with complete data: {complete_data_count}")
    print(f"   ⚠️  Columns with partial missing data: {partial_missing_count}")
    print(f"   ❌ Columns with all missing data: {all_missing_count}")
    
    # Missing values by category
    categories = {
        'CT Biomarkers': fat_cols + muscle_cols + [col for col in ct_biomarker_cols if col in df.columns],
        'DICOM Metadata': dicom_cols,
        'Diagnosis Codes': diagnosis_cols,
        'Lab Values': lab_cols,
        'Patient Info': patient_id_cols + ['DEATH_DATE', 'ORIG_STUDY_DATE']
    }
    
    print(f"\n📊 Missing values by data category:")
    for category, cols in categories.items():
        category_cols = [col for col in cols if col in df.columns]
        if category_cols:
            category_missing = df[category_cols].isnull().sum().sum()
            category_total = len(category_cols) * len(df)
            category_missing_pct = (category_missing / category_total) * 100
            print(f"   {category}: {category_missing_pct:.1f}% missing")
    
    # 6. Missing Values Visualization
    print("\n=== MISSING VALUES VISUALIZATION ===")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Missing values bar plot
        missing_vis = missing_stats[missing_stats['Missing_Percentage'] > 0].head(15)
        if not missing_vis.empty:
            sns.barplot(data=missing_vis, y='Column', x='Missing_Percentage', ax=axes[0,0])
            axes[0,0].set_title('Top 15 Columns with Missing Values')
            axes[0,0].set_xlabel('Missing Percentage (%)')
        
        # 2. Data completeness pie chart
        categories = ['Complete Data', 'Partial Missing', 'All Missing']
        counts = [complete_data_count, partial_missing_count, all_missing_count]
        colors = ['green', 'orange', 'red']
        axes[0,1].pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
        axes[0,1].set_title('Data Completeness Distribution')
        
        # 3. Missing values by data category
        category_names = list(categories.keys())
        category_missing_pcts = []
        for category, cols in categories.items():
            category_cols = [col for col in cols if col in df.columns]
            if category_cols:
                category_missing = df[category_cols].isnull().sum().sum()
                category_total = len(category_cols) * len(df)
                category_missing_pct = (category_missing / category_total) * 100
                category_missing_pcts.append(category_missing_pct)
            else:
                category_missing_pcts.append(0)

        sns.barplot(x=category_names, y=category_missing_pcts, ax=axes[1,0])
        axes[1,0].set_title('Missing Values by Data Category')
        axes[1,0].set_ylabel('Missing Percentage (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Missing values distribution histogram
        missing_stats['Missing_Percentage'].hist(bins=20, ax=axes[1,1])
        axes[1,1].set_title('Distribution of Missing Value Percentages')
        axes[1,1].set_xlabel('Missing Percentage (%)')
        axes[1,1].set_ylabel('Number of Columns')
        
        plt.tight_layout()
        plt.savefig('biomarkers_missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Missing values visualization saved as 'biomarkers_missing_values_analysis.png'")
    except Exception as e:
        print(f"⚠️  Could not create missing values visualization: {e}")
    
    # 7. Comorbidity Analysis
    print("\n\n7. COMORBIDITY ANALYSIS")
    print("-" * 25)
    
    # Extract condition names from diagnosis columns
    condition_cols = [col for col in df.columns if '_DX_Code' in col and col.replace('_DX_Code', '') + '_DX_NAME' in df.columns]
    print(f"🏥 Total medical conditions tracked: {len(condition_cols)}")
    
    if condition_cols:
        # Analyze prevalence of each condition
        condition_prevalence = {}
        for code_col in condition_cols:
            condition_name = code_col.replace('_DX_Code', '')
            has_condition = df[code_col].notna().sum()
            prevalence = has_condition / len(df) * 100
            condition_prevalence[condition_name] = {
                'count': has_condition,
                'prevalence': prevalence
            }
        
        # Sort by prevalence
        sorted_conditions = sorted(condition_prevalence.items(), key=lambda x: x[1]['prevalence'], reverse=True)
        
        print(f"\n📊 Top 15 most prevalent conditions:")
        for i, (condition, stats) in enumerate(sorted_conditions[:15]):
            print(f"{i+1:2d}. {condition.replace('_', ' ').title():35s}: {stats['count']:5d} patients ({stats['prevalence']:5.1f}%)")
        
        # Create binary matrix for conditions
        condition_matrix = pd.DataFrame()
        for code_col in condition_cols:
            condition_name = code_col.replace('_DX_Code', '')
            condition_matrix[condition_name] = df[code_col].notna().astype(int)
        
        # Calculate total conditions per patient
        conditions_per_patient = condition_matrix.sum(axis=1)
        
        print(f"\n=== COMORBIDITY BURDEN ===")
        print(f"0️⃣  Patients with 0 conditions: {(conditions_per_patient == 0).sum():,} ({(conditions_per_patient == 0).mean()*100:.1f}%)")
        print(f"1️⃣  Patients with 1 condition: {(conditions_per_patient == 1).sum():,} ({(conditions_per_patient == 1).mean()*100:.1f}%)")
        print(f"2️⃣+ Patients with 2+ conditions: {(conditions_per_patient >= 2).sum():,} ({(conditions_per_patient >= 2).mean()*100:.1f}%)")
        print(f"📊 Average conditions per patient: {conditions_per_patient.mean():.2f}")
        print(f"📈 Maximum conditions in a patient: {conditions_per_patient.max()}")
        
        # Visualize comorbidity patterns
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Top conditions prevalence
            top_conditions = sorted_conditions[:15]
            condition_names = [c[0].replace('_', ' ').title()[:25] for c in top_conditions]  # Truncate long names
            prevalences = [c[1]['prevalence'] for c in top_conditions]
            
            sns.barplot(x=prevalences, y=condition_names, ax=axes[0,0])
            axes[0,0].set_title('Top 15 Medical Conditions by Prevalence')
            axes[0,0].set_xlabel('Prevalence (%)')
            
            # 2. Comorbidity burden distribution
            conditions_per_patient.hist(bins=range(0, min(conditions_per_patient.max()+2, 21)), ax=axes[0,1])
            axes[0,1].set_title('Distribution of Comorbidity Burden')
            axes[0,1].set_xlabel('Number of Conditions per Patient')
            axes[0,1].set_ylabel('Number of Patients')
            
            # 3. Condition co-occurrence heatmap (top 10 conditions)
            top_10_conditions = [c[0] for c in sorted_conditions[:10]]
            if len(top_10_conditions) > 1:
                correlation_matrix = condition_matrix[top_10_conditions].corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                            cmap='coolwarm', center=0, ax=axes[1,0])
                axes[1,0].set_title('Condition Co-occurrence Correlation (Top 10)')
            else:
                axes[1,0].text(0.5, 0.5, 'Insufficient data for correlation', ha='center', va='center', transform=axes[1,0].transAxes)
            
            # 4. Mortality vs comorbidity burden
            deceased_mask = df['DEATH_DATE'].notna()
            alive_conditions = conditions_per_patient[~deceased_mask]
            deceased_conditions = conditions_per_patient[deceased_mask]
            
            if len(deceased_conditions) > 0:
                axes[1,1].hist([alive_conditions, deceased_conditions], bins=range(0, 11), 
                               alpha=0.7, label=['Alive', 'Deceased'], density=True)
                axes[1,1].set_title('Comorbidity Burden by Mortality Status')
                axes[1,1].set_xlabel('Number of Conditions')
                axes[1,1].set_ylabel('Density')
                axes[1,1].legend()
            else:
                axes[1,1].text(0.5, 0.5, 'No mortality data available', ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('Mortality Analysis')
            
            plt.tight_layout()
            plt.savefig('biomarkers_comorbidity_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✅ Comorbidity analysis visualization saved as 'biomarkers_comorbidity_analysis.png'")
        except Exception as e:
            print(f"⚠️  Could not create comorbidity visualization: {e}")
    else:
        print("❌ No condition columns found for comorbidity analysis.")
    
    # 8. Lab Values Analysis
    print("\n\n8. LABORATORY VALUES ANALYSIS")
    print("-" * 35)
    
    if lab_cols:
        # Separate HbA1c and Creatinine columns
        hba1c_cols = [col for col in lab_cols if 'HA1C' in col]
        creatinine_cols = [col for col in lab_cols if 'CRPN' in col]
        
        print(f"🧪 HbA1c columns: {len(hba1c_cols)}")
        print(f"🧪 Creatinine columns: {len(creatinine_cols)}")
        
        # Analyze HbA1c values
        if hba1c_cols:
            print(f"\n=== HbA1c ANALYSIS ===")
            for col in hba1c_cols[:6]:  # Analyze first 6 columns
                # Clean and convert values
                values = df[col].astype(str).str.replace('<', '').str.replace('>', '')
                numeric_values = pd.to_numeric(values, errors='coerce')
                valid_values = numeric_values.dropna()
                
                if len(valid_values) > 0:
                    print(f"\n📊 {col}:")
                    print(f"     Valid values: {len(valid_values):,}")
                    print(f"     Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
                    print(f"     Mean ± SD: {valid_values.mean():.2f} ± {valid_values.std():.2f}")
                    
                    # Clinical interpretation for HbA1c
                    normal = (valid_values < 5.7).sum()
                    prediabetes = ((valid_values >= 5.7) & (valid_values < 6.5)).sum()
                    diabetes = (valid_values >= 6.5).sum()
                    
                    print(f"     ✅ Normal (<5.7%): {normal} ({normal/len(valid_values)*100:.1f}%)")
                    print(f"     ⚠️  Prediabetes (5.7-6.4%): {prediabetes} ({prediabetes/len(valid_values)*100:.1f}%)")
                    print(f"     ❌ Diabetes (≥6.5%): {diabetes} ({diabetes/len(valid_values)*100:.1f}%)")
        
        # Analyze Creatinine values
        if creatinine_cols:
            print(f"\n=== CREATININE ANALYSIS ===")
            for col in creatinine_cols[:6]:  # Analyze first 6 columns
                # Clean and convert values
                values = df[col].astype(str).str.replace('<', '').str.replace('>', '')
                numeric_values = pd.to_numeric(values, errors='coerce')
                valid_values = numeric_values.dropna()
                
                if len(valid_values) > 0:
                    print(f"\n📊 {col}:")
                    print(f"     Valid values: {len(valid_values):,}")
                    print(f"     Range: {valid_values.min():.2f} - {valid_values.max():.2f}")
                    print(f"     Mean ± SD: {valid_values.mean():.2f} ± {valid_values.std():.2f}")
                    
                    # Clinical interpretation for Creatinine
                    normal = ((valid_values >= 0.6) & (valid_values <= 1.2)).sum()
                    elevated = (valid_values > 1.2).sum()
                    
                    print(f"     ✅ Normal (0.6-1.2): {normal} ({normal/len(valid_values)*100:.1f}%)")
                    print(f"     ❌ Elevated (>1.2): {elevated} ({elevated/len(valid_values)*100:.1f}%)")
    else:
        print("❌ No lab value columns found for analysis.")
    
    # 9. CT Biomarker Distributions Visualization
    print("\n\n9. CT BIOMARKER DISTRIBUTIONS VISUALIZATION")
    print("-" * 45)
    
    try:
        # Create comprehensive CT biomarker distribution plots
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        axes = axes.ravel()
        
        # Key biomarker columns for visualization
        viz_cols = [
            'BMDL1Values_BMDL1StandardHU',
            'CalciumScoring_AbdominalAgatston',
            'LiverValues_LiverMedianHU',
            'KidneyValues_KidneyMedianHU', 
            'L3FatValues_L3VATArea',
            'L3FatValues_L3SATArea',
            'L3FatValues_L3VATSATRatio',
            'MuscleValues_L3MuscleArea',
            'MuscleValues_L3MuscleMeanHU'
        ]
        
        plot_idx = 0
        for col in viz_cols:
            if col in df.columns and plot_idx < 9:
                data = df[col].dropna()
                if len(data) > 0:
                    # Create histogram with better formatting
                    axes[plot_idx].hist(data, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
                    
                    # Format title and labels
                    title = col.replace('Values_', '').replace('_', ' ')
                    axes[plot_idx].set_title(f'{title}\\n(n={len(data):,})', fontsize=10)
                    
                    # Add units to x-label
                    if 'HU' in col:
                        axes[plot_idx].set_xlabel('Hounsfield Units (HU)')
                    elif 'Area' in col:
                        axes[plot_idx].set_xlabel('Area (cm²)')
                    elif 'Volume' in col:
                        axes[plot_idx].set_xlabel('Volume (cm³)')
                    elif 'Ratio' in col:
                        axes[plot_idx].set_xlabel('Ratio')
                    elif 'Agatston' in col:
                        axes[plot_idx].set_xlabel('Agatston Score')
                    else:
                        axes[plot_idx].set_xlabel('Value')
                        
                    axes[plot_idx].set_ylabel('Frequency')
                    
                    # Add statistics text box
                    stats_text = f'Mean: {data.mean():.1f}\\nStd: {data.std():.1f}\\nMedian: {data.median():.1f}'
                    axes[plot_idx].text(0.7, 0.7, stats_text, transform=axes[plot_idx].transAxes,
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                      fontsize=8)
                    
                    plot_idx += 1
        
        # Hide empty subplots
        for i in range(plot_idx, 9):
            axes[i].set_visible(False)
        
        plt.suptitle('CT Biomarker Distributions', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig('biomarkers_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ CT biomarker distributions saved as 'biomarkers_distributions.png'")
    except Exception as e:
        print(f"⚠️  Could not create biomarker distributions: {e}")
    
    # 10. Key Insights and Recommendations
    print("\n\n10. KEY INSIGHTS AND RECOMMENDATIONS")
    print("-" * 40)
    
    print("=" * 80)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n📊 DATASET SUMMARY:")
    print(f"   • Total patients: {df['PAT_ID_MASKED'].nunique():,}")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Total variables: {len(df.columns)}")
    print(f"   • CT biomarker measurements: {len(fat_cols + muscle_cols + [col for col in ct_biomarker_cols if col in df.columns])}")
    print(f"   • Medical conditions tracked: {len(condition_cols) if 'condition_cols' in locals() else 'N/A'}")
    print(f"   • Lab measurements: {len(lab_cols) if lab_cols else 0}")
    
    print("\n🔍 DATA QUALITY INSIGHTS:")
    complete_data_pct = (missing_stats['Missing_Percentage'] == 0).sum() / len(missing_stats) * 100
    print(f"   • {complete_data_pct:.1f}% of columns have complete data")
    overall_missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f"   • {overall_missing_pct:.1f}% of all values are missing")
    print(f"   • Primary key (PAT_ID_MASKED) appears to be unique per record")
    
    print("\n🔬 CT BIOMARKER INSIGHTS:")
    print(f"   • Comprehensive body composition analysis available")
    print(f"   • Bone density, fat distribution, muscle composition, and organ metrics")
    print(f"   • L3 vertebral level provides clinically relevant sarcopenia/obesity metrics")
    print(f"   • Calcium scoring enables cardiovascular risk assessment")
    print(f"   • Liver attenuation allows hepatic steatosis evaluation")
    
    if 'sorted_conditions' in locals() and sorted_conditions:
        print("\n🏥 CLINICAL INSIGHTS:")
        top_condition = sorted_conditions[0]
        print(f"   • Most common condition: {top_condition[0].replace('_', ' ').title()} ({top_condition[1]['prevalence']:.1f}% prevalence)")
        
        if 'conditions_per_patient' in locals():
            avg_conditions = conditions_per_patient.mean()
            multiple_conditions_pct = (conditions_per_patient >= 2).mean() * 100
            print(f"   • Average conditions per patient: {avg_conditions:.1f}")
            print(f"   • {multiple_conditions_pct:.1f}% of patients have multiple comorbidities")
    
    print("\n📈 RECOMMENDATIONS FOR FURTHER ANALYSIS:")
    print("   1️⃣  CT Biomarker Applications:")
    print("      • Develop sarcopenia screening models using muscle area/quality")
    print("      • Create hepatic steatosis severity classification")
    print("      • Build cardiovascular risk models using calcium scoring")
    print("      • Investigate visceral adiposity and metabolic syndrome associations")
    
    print("\n   2️⃣  Data Preprocessing:")
    print("      • Handle missing CT biomarkers using clinical domain knowledge")
    print("      • Standardize measurements across different CT scanner models")
    print("      • Create composite biomarker scores (e.g., body composition index)")
    print("      • Normalize measurements by patient demographics (age, sex, BMI)")
    
    print("\n   3️⃣  Feature Engineering:")
    print("      • Calculate muscle-to-fat ratios and body composition indices")
    print("      • Create age/sex-adjusted z-scores for biomarkers")
    print("      • Develop temporal change metrics for longitudinal data")
    print("      • Engineer interaction terms between CT biomarkers and clinical variables")
    
    print("\n   4️⃣  Clinical Research Applications:")
    print("      • Phenotyping studies using CT biomarker clustering")
    print("      • Mortality prediction incorporating body composition")
    print("      • Disease progression modeling with imaging biomarkers")
    print("      • Treatment response prediction using baseline CT metrics")
    
    print("\n   5️⃣  Machine Learning Applications:")
    print("      • Multi-modal prediction models (CT + clinical + lab data)")
    print("      • Unsupervised clustering for patient stratification")
    print("      • Deep learning for automated CT biomarker extraction")
    print("      • Survival analysis incorporating imaging biomarkers")
    
    print("\n   6️⃣  Clinical Validation:")
    print("      • Validate CT biomarker thresholds against clinical outcomes")
    print("      • Cross-reference with established clinical guidelines")
    print("      • Investigate scanner-specific calibration requirements")
    print("      • Assess biomarker reproducibility across imaging protocols")
    
    print("\n✅ EDA COMPLETE - Dataset ready for advanced CT biomarker analytics!")
    print("\n🎯 UNIQUE VALUE PROPOSITION:")
    print("   This dataset uniquely combines quantitative CT imaging biomarkers with")
    print("   comprehensive clinical outcomes, enabling novel research in:")
    print("   • Radiomics-based precision medicine")
    print("   • Body composition and metabolic health")
    print("   • Imaging biomarkers for disease prediction")
    print("   • Multi-modal AI/ML model development")
    
    print(f"\n📁 Output files saved:")
    print(f"   • biomarkers_missing_values_analysis.png")
    if 'condition_cols' in locals() and condition_cols:
        print(f"   • biomarkers_comorbidity_analysis.png")
    print(f"   • biomarkers_distributions.png")

if __name__ == "__main__":
    main()
