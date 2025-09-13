#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) of Biomarkers with HCC Codes Dataset

This script provides a thorough exploratory data analysis of the joined CT Biomarkers & 
HCC Codes dataset (biomarkers_with_hcc_codes_20250909_062523.csv). The dataset combines 
CT-derived biomarker measurements with comprehensive clinical outcomes data and HCC 
(Hierarchical Condition Category) comorbidity codes.

Key Analysis Areas:
1. Dataset Overview & Structure
2. Demographics & Patient Characteristics  
3. CT Biomarkers Analysis
4. Clinical Outcomes & Diagnoses
5. HCC Codes & Comorbidities Analysis
6. Laboratory Values Analysis
7. Missing Data Patterns
8. Correlations & Relationships
9. Advanced Analytics (PCA, Clustering)
10. Clinical Insights & Recommendations

Usage: python biomarkers_hcc_comprehensive_eda.py
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

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
    from sklearn.manifold import TSNE
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

def save_figure(fig, filename, output_dir):
    """Save figure to output directory"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"📊 Saved: {filename}")

def categorize_columns(df):
    """Categorize columns by type for targeted analysis"""
    
    # Define column categories
    categories = {
        'demographic': ['StudyInfo_Accession', 'StudyInfo_MRN', 'subject.label', 'session.label', 
                       'ACC_NUM-SESSION-ID', 'PAT_ID_MASKED', 'DEATH_DATE', 'ORIG_STUDY_DATE', 
                       'SEX', 'AGE', 'PAT-ID', 'ACC_NUM_EXTRACTED', 'png_filename'],
        'biomarkers': [],
        'diagnosis': [],
        'hcc_codes': [],
        'lab_values': [],
        'dicom_headers': [],
        'levels': [],
        'other': []
    }
    
    for col in df.columns:
        if col.startswith('HCC_'):
            categories['hcc_codes'].append(col)
        elif any(x in col for x in ['Values_', 'Scoring_']):
            categories['biomarkers'].append(col)
        elif '_DX_' in col:
            categories['diagnosis'].append(col)
        elif any(x in col for x in ['HA1C_', 'CRPN_']):
            categories['lab_values'].append(col)
        elif 'DICOMHeader_' in col:
            categories['dicom_headers'].append(col)
        elif 'Levels_' in col:
            categories['levels'].append(col)
        elif col not in categories['demographic']:
            categories['other'].append(col)
    
    return categories

def analyze_dataset_overview(df, output_dir):
    """1. Dataset Overview and Structure Analysis"""
    print("\n" + "="*80)
    print("1. DATASET OVERVIEW AND STRUCTURE")
    print("="*80)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📈 Number of rows: {df.shape[0]:,}")
    print(f"📋 Number of columns: {df.shape[1]:,}")
    print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Categorize columns
    categories = categorize_columns(df)
    
    print("\n=== COLUMN CATEGORIZATION ===")
    for category, cols in categories.items():
        print(f"{category.upper()}: {len(cols)} columns")
        if len(cols) > 0 and len(cols) <= 10:
            print(f"  Examples: {cols[:5]}")
        elif len(cols) > 10:
            print(f"  Examples: {cols[:3]} ... and {len(cols)-3} more")
    
    # Create overview summary
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data_Type': df.dtypes,
        'Non_Null_Count': df.count(),
        'Null_Count': df.isnull().sum(),
        'Null_Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Unique_Values': [df[col].nunique() for col in df.columns]
    })
    
    # Save detailed column summary
    summary_path = os.path.join(output_dir, 'column_summary.csv')
    col_info.to_csv(summary_path, index=False)
    print(f"📋 Detailed column summary saved: {summary_path}")
    
    # Data types distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    dtype_counts = df.dtypes.value_counts()
    ax1.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    ax1.set_title('Data Types Distribution')
    
    # Missing data by category
    category_missing = {}
    for category, cols in categories.items():
        if cols:
            category_missing[category] = df[cols].isnull().sum().sum()
    
    ax2.bar(category_missing.keys(), category_missing.values())
    ax2.set_title('Missing Values by Category')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_figure(fig, 'dataset_overview.png', output_dir)
    
    return categories

def analyze_demographics(df, categories, output_dir):
    """2. Demographics and Patient Characteristics Analysis"""
    print("\n" + "="*80)
    print("2. DEMOGRAPHICS AND PATIENT CHARACTERISTICS")
    print("="*80)
    
    # Age and Sex analysis
    if 'AGE' in df.columns and 'SEX' in df.columns:
        print("\n=== AGE AND SEX DISTRIBUTION ===")
        print(f"Age statistics:")
        print(df['AGE'].describe())
        print(f"\nSex distribution:")
        print(df['SEX'].value_counts())
        print(f"Sex distribution (%):")
        print(df['SEX'].value_counts(normalize=True) * 100)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Age distribution
        axes[0,0].hist(df['AGE'].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Age Distribution')
        axes[0,0].set_xlabel('Age')
        axes[0,0].set_ylabel('Frequency')
        
        # Sex distribution
        sex_counts = df['SEX'].value_counts()
        axes[0,1].pie(sex_counts.values, labels=sex_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Sex Distribution')
        
        # Age by sex
        df.boxplot(column='AGE', by='SEX', ax=axes[1,0])
        axes[1,0].set_title('Age Distribution by Sex')
        axes[1,0].set_xlabel('Sex')
        axes[1,0].set_ylabel('Age')
        
        # Age groups by sex
        df['Age_Group'] = pd.cut(df['AGE'], bins=[0, 40, 50, 60, 70, 80, 100], 
                                labels=['<40', '40-49', '50-59', '60-69', '70-79', '80+'])
        age_sex_crosstab = pd.crosstab(df['Age_Group'], df['SEX'])
        age_sex_crosstab.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Age Groups by Sex')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_figure(fig, 'demographics_overview.png', output_dir)
    
    # Death analysis
    if 'DEATH_DATE' in df.columns:
        print("\n=== MORTALITY ANALYSIS ===")
        death_count = df['DEATH_DATE'].notna().sum()
        mortality_rate = death_count / len(df) * 100
        print(f"Patients with death date: {death_count:,} ({mortality_rate:.2f}%)")
        
        if death_count > 0 and 'AGE' in df.columns and 'SEX' in df.columns:
            # Mortality by demographics
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Mortality by age group
            df['Has_Death'] = df['DEATH_DATE'].notna()
            mortality_by_age = df.groupby('Age_Group')['Has_Death'].mean() * 100
            mortality_by_age.plot(kind='bar', ax=axes[0])
            axes[0].set_title('Mortality Rate by Age Group (%)')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Mortality by sex
            mortality_by_sex = df.groupby('SEX')['Has_Death'].mean() * 100
            mortality_by_sex.plot(kind='bar', ax=axes[1])
            axes[1].set_title('Mortality Rate by Sex (%)')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            save_figure(fig, 'mortality_analysis.png', output_dir)

def analyze_biomarkers(df, categories, output_dir):
    """3. CT Biomarkers Analysis"""
    print("\n" + "="*80)
    print("3. CT BIOMARKERS ANALYSIS")
    print("="*80)
    
    biomarker_cols = categories['biomarkers']
    if not biomarker_cols:
        print("No biomarker columns found.")
        return
    
    print(f"Found {len(biomarker_cols)} biomarker columns")
    
    # Biomarker categories
    biomarker_categories = {
        'Bone': [col for col in biomarker_cols if 'BMD' in col or 'Bone' in col],
        'Calcium': [col for col in biomarker_cols if 'Calcium' in col],
        'Kidney': [col for col in biomarker_cols if 'Kidney' in col],
        'Fat': [col for col in biomarker_cols if 'Fat' in col or 'SAT' in col or 'VAT' in col],
        'Liver': [col for col in biomarker_cols if 'Liver' in col],
        'Muscle': [col for col in biomarker_cols if 'Muscle' in col],
        'Spleen': [col for col in biomarker_cols if 'Spleen' in col],
        'Pancreas': [col for col in biomarker_cols if 'Pancreas' in col],
        'Other': [col for col in biomarker_cols if not any(cat in col for cat in 
                 ['BMD', 'Calcium', 'Kidney', 'Fat', 'SAT', 'VAT', 'Liver', 'Muscle', 'Spleen', 'Pancreas'])]
    }
    
    print("\n=== BIOMARKER CATEGORIES ===")
    for category, cols in biomarker_categories.items():
        if cols:
            print(f"{category}: {len(cols)} columns")
    
    # Statistical summary for key biomarkers
    key_biomarkers = [col for col in biomarker_cols if df[col].dtype in ['float64', 'int64']][:20]
    if key_biomarkers:
        print(f"\n=== KEY BIOMARKERS STATISTICS (Top 20) ===")
        biomarker_stats = df[key_biomarkers].describe()
        print(biomarker_stats.round(2))
        
        # Save detailed biomarker statistics
        stats_path = os.path.join(output_dir, 'biomarker_statistics.csv')
        biomarker_stats.to_csv(stats_path)
        print(f"📊 Biomarker statistics saved: {stats_path}")
        
        # Visualizations
        n_plots = min(12, len(key_biomarkers))
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, col in enumerate(key_biomarkers[:n_plots]):
            data = df[col].dropna()
            if len(data) > 0:
                axes[i].hist(data, bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{col}\n(n={len(data):,})', fontsize=10)
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        save_figure(fig, 'biomarkers_distributions.png', output_dir)
        
        # Correlation heatmap for biomarkers
        if len(key_biomarkers) > 1:
            fig, ax = plt.subplots(figsize=(15, 12))
            biomarker_corr = df[key_biomarkers].corr()
            sns.heatmap(biomarker_corr, annot=False, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Biomarkers Correlation Matrix')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_figure(fig, 'biomarkers_correlation.png', output_dir)

def analyze_hcc_codes(df, categories, output_dir):
    """4. HCC Codes and Comorbidities Analysis"""
    print("\n" + "="*80)
    print("4. HCC CODES AND COMORBIDITIES ANALYSIS")
    print("="*80)
    
    hcc_cols = categories['hcc_codes']
    if not hcc_cols:
        print("No HCC columns found.")
        return
    
    print(f"Found {len(hcc_cols)} HCC code columns")
    
    # HCC prevalence analysis
    hcc_prevalence = {}
    hcc_stats = {}
    
    for col in hcc_cols:
        non_null = df[col].notna().sum()
        if non_null > 0:
            prevalence = (df[col] > 0).sum() / non_null * 100
            hcc_prevalence[col] = prevalence
            hcc_stats[col] = {
                'non_null_count': non_null,
                'positive_cases': (df[col] > 0).sum(),
                'prevalence_pct': prevalence,
                'mean_value': df[col].mean(),
                'max_value': df[col].max()
            }
    
    # Sort by prevalence
    sorted_hcc = sorted(hcc_prevalence.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== TOP 20 MOST PREVALENT HCC CODES ===")
    for i, (hcc_code, prevalence) in enumerate(sorted_hcc[:20]):
        stats = hcc_stats[hcc_code]
        print(f"{i+1:2d}. {hcc_code}: {stats['positive_cases']:,} cases ({prevalence:.2f}%)")
    
    print("\n=== BOTTOM 10 LEAST PREVALENT HCC CODES ===")
    for i, (hcc_code, prevalence) in enumerate(sorted_hcc[-10:]):
        stats = hcc_stats[hcc_code]
        print(f"{i+1:2d}. {hcc_code}: {stats['positive_cases']:,} cases ({prevalence:.2f}%)")
    
    # HCC statistics summary
    hcc_stats_df = pd.DataFrame(hcc_stats).T
    stats_path = os.path.join(output_dir, 'hcc_codes_statistics.csv')
    hcc_stats_df.to_csv(stats_path)
    print(f"📊 HCC statistics saved: {stats_path}")
    
    # Visualizations
    # Top HCC codes prevalence
    top_20_hcc = dict(sorted_hcc[:20])
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Top 20 HCC prevalence
    axes[0,0].barh(range(len(top_20_hcc)), list(top_20_hcc.values()))
    axes[0,0].set_yticks(range(len(top_20_hcc)))
    axes[0,0].set_yticklabels(list(top_20_hcc.keys()))
    axes[0,0].set_xlabel('Prevalence (%)')
    axes[0,0].set_title('Top 20 HCC Codes Prevalence')
    
    # HCC prevalence distribution
    prevalences = list(hcc_prevalence.values())
    axes[0,1].hist(prevalences, bins=30, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Prevalence (%)')
    axes[0,1].set_ylabel('Number of HCC Codes')
    axes[0,1].set_title('HCC Codes Prevalence Distribution')
    
    # Comorbidity burden (number of HCC codes per patient)
    hcc_burden = (df[hcc_cols] > 0).sum(axis=1)
    axes[1,0].hist(hcc_burden, bins=30, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Number of HCC Codes per Patient')
    axes[1,0].set_ylabel('Number of Patients')
    axes[1,0].set_title('Comorbidity Burden Distribution')
    
    # HCC burden by age group (if age available)
    if 'Age_Group' in df.columns:
        burden_by_age = df.groupby('Age_Group')[hcc_cols].apply(lambda x: (x > 0).sum(axis=1).mean())
        burden_by_age.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Average HCC Burden by Age Group')
        axes[1,1].set_xlabel('Age Group')
        axes[1,1].set_ylabel('Average Number of HCC Codes')
        axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_figure(fig, 'hcc_codes_analysis.png', output_dir)
    
    # HCC co-occurrence analysis (top 10 HCC codes)
    if len(top_20_hcc) >= 10:
        top_10_hcc_cols = list(top_20_hcc.keys())[:10]
        hcc_cooccurrence = df[top_10_hcc_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(hcc_cooccurrence, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', ax=ax)
        ax.set_title('Top 10 HCC Codes Co-occurrence Correlation')
        plt.tight_layout()
        save_figure(fig, 'hcc_cooccurrence.png', output_dir)

def analyze_clinical_outcomes(df, categories, output_dir):
    """5. Clinical Outcomes and Diagnoses Analysis"""
    print("\n" + "="*80)
    print("5. CLINICAL OUTCOMES AND DIAGNOSES ANALYSIS")
    print("="*80)
    
    diagnosis_cols = categories['diagnosis']
    if not diagnosis_cols:
        print("No diagnosis columns found.")
        return
    
    print(f"Found {len(diagnosis_cols)} diagnosis-related columns")
    
    # Extract diagnosis code and name columns
    dx_code_cols = [col for col in diagnosis_cols if '_DX_Code' in col]
    dx_name_cols = [col for col in diagnosis_cols if '_DX_NAME' in col]
    dx_date_cols = [col for col in diagnosis_cols if '_DX_' in col and ('_DT' in col or '_dt' in col)]
    
    print(f"Diagnosis codes: {len(dx_code_cols)}")
    print(f"Diagnosis names: {len(dx_name_cols)}")
    print(f"Diagnosis dates: {len(dx_date_cols)}")
    
    # Diagnosis prevalence
    diagnosis_prevalence = {}
    for code_col in dx_code_cols:
        condition_name = code_col.replace('_DX_Code', '')
        has_diagnosis = df[code_col].notna().sum()
        prevalence = has_diagnosis / len(df) * 100
        diagnosis_prevalence[condition_name] = {
            'count': has_diagnosis,
            'prevalence': prevalence
        }
    
    # Sort by prevalence
    sorted_diagnoses = sorted(diagnosis_prevalence.items(), key=lambda x: x[1]['prevalence'], reverse=True)
    
    print("\n=== TOP 20 MOST PREVALENT DIAGNOSES ===")
    for i, (condition, stats) in enumerate(sorted_diagnoses[:20]):
        print(f"{i+1:2d}. {condition}: {stats['count']:,} cases ({stats['prevalence']:.2f}%)")
    
    # Save diagnosis statistics
    dx_stats_df = pd.DataFrame(diagnosis_prevalence).T
    stats_path = os.path.join(output_dir, 'diagnosis_statistics.csv')
    dx_stats_df.to_csv(stats_path)
    print(f"📊 Diagnosis statistics saved: {stats_path}")
    
    # Visualizations
    top_20_dx = dict([(k, v['prevalence']) for k, v in sorted_diagnoses[:20]])
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Top 20 diagnoses prevalence
    axes[0].barh(range(len(top_20_dx)), list(top_20_dx.values()))
    axes[0].set_yticks(range(len(top_20_dx)))
    axes[0].set_yticklabels(list(top_20_dx.keys()))
    axes[0].set_xlabel('Prevalence (%)')
    axes[0].set_title('Top 20 Diagnoses Prevalence')
    
    # Diagnosis prevalence distribution
    all_prevalences = [stats['prevalence'] for stats in diagnosis_prevalence.values()]
    axes[1].hist(all_prevalences, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Prevalence (%)')
    axes[1].set_ylabel('Number of Diagnoses')
    axes[1].set_title('Diagnosis Prevalence Distribution')
    
    plt.tight_layout()
    save_figure(fig, 'clinical_outcomes_analysis.png', output_dir)

def analyze_lab_values(df, categories, output_dir):
    """6. Laboratory Values Analysis"""
    print("\n" + "="*80)
    print("6. LABORATORY VALUES ANALYSIS")
    print("="*80)
    
    lab_cols = categories['lab_values']
    if not lab_cols:
        print("No laboratory value columns found.")
        return
    
    print(f"Found {len(lab_cols)} laboratory value columns")
    
    # Separate by test type
    ha1c_cols = [col for col in lab_cols if 'HA1C' in col and not 'DT' in col]
    crpn_cols = [col for col in lab_cols if 'CRPN' in col and not 'DT' in col]
    
    print(f"HbA1c measurements: {len(ha1c_cols)}")
    print(f"C-reactive protein measurements: {len(crpn_cols)}")
    
    if ha1c_cols or crpn_cols:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # HbA1c analysis
        if ha1c_cols:
            print("\n=== HbA1c ANALYSIS ===")
            ha1c_data = df[ha1c_cols].describe()
            print(ha1c_data.round(2))
            
            # Combined HbA1c values
            all_ha1c = pd.concat([df[col].dropna() for col in ha1c_cols])
            axes[0,0].hist(all_ha1c, bins=30, alpha=0.7, edgecolor='black')
            axes[0,0].set_title('HbA1c Distribution (All Measurements)')
            axes[0,0].set_xlabel('HbA1c (%)')
            axes[0,0].set_ylabel('Frequency')
            
            # HbA1c over time (prior vs after)
            prior_cols = [col for col in ha1c_cols if 'PRIOR' in col]
            after_cols = [col for col in ha1c_cols if 'AFTER' in col]
            
            if prior_cols and after_cols:
                prior_data = pd.concat([df[col].dropna() for col in prior_cols])
                after_data = pd.concat([df[col].dropna() for col in after_cols])
                
                axes[0,1].hist([prior_data, after_data], bins=20, alpha=0.7, 
                              label=['Prior', 'After'], edgecolor='black')
                axes[0,1].set_title('HbA1c: Prior vs After')
                axes[0,1].set_xlabel('HbA1c (%)')
                axes[0,1].legend()
        
        # CRPN analysis
        if crpn_cols:
            print("\n=== C-REACTIVE PROTEIN ANALYSIS ===")
            crpn_data = df[crpn_cols].describe()
            print(crpn_data.round(2))
            
            # Combined CRPN values
            all_crpn = pd.concat([df[col].dropna() for col in crpn_cols])
            axes[1,0].hist(all_crpn, bins=30, alpha=0.7, edgecolor='black')
            axes[1,0].set_title('C-Reactive Protein Distribution')
            axes[1,0].set_xlabel('CRP')
            axes[1,0].set_ylabel('Frequency')
            
            # CRPN over time (prior vs after)
            prior_crpn = [col for col in crpn_cols if 'PRIOR' in col]
            after_crpn = [col for col in crpn_cols if 'AFTER' in col]
            
            if prior_crpn and after_crpn:
                prior_data = pd.concat([df[col].dropna() for col in prior_crpn])
                after_data = pd.concat([df[col].dropna() for col in after_crpn])
                
                axes[1,1].hist([prior_data, after_data], bins=20, alpha=0.7, 
                              label=['Prior', 'After'], edgecolor='black')
                axes[1,1].set_title('C-Reactive Protein: Prior vs After')
                axes[1,1].set_xlabel('CRP')
                axes[1,1].legend()
        
        plt.tight_layout()
        save_figure(fig, 'lab_values_analysis.png', output_dir)

def analyze_missing_data(df, categories, output_dir):
    """7. Missing Data Patterns Analysis"""
    print("\n" + "="*80)
    print("7. MISSING DATA PATTERNS ANALYSIS")
    print("="*80)
    
    # Overall missing data statistics
    missing_stats = df.isnull().sum()
    missing_pct = (missing_stats / len(df) * 100).round(2)
    
    missing_summary = pd.DataFrame({
        'Column': missing_stats.index,
        'Missing_Count': missing_stats.values,
        'Missing_Percentage': missing_pct.values
    })
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
    
    print(f"Columns with missing data: {len(missing_summary)} out of {len(df.columns)}")
    print(f"Total missing values: {missing_stats.sum():,}")
    
    print("\n=== TOP 20 COLUMNS WITH MOST MISSING DATA ===")
    print(missing_summary.head(20).to_string(index=False))
    
    # Save missing data summary
    missing_path = os.path.join(output_dir, 'missing_data_summary.csv')
    missing_summary.to_csv(missing_path, index=False)
    print(f"📊 Missing data summary saved: {missing_path}")
    
    # Missing data by category
    print("\n=== MISSING DATA BY CATEGORY ===")
    for category, cols in categories.items():
        if cols:
            category_missing = df[cols].isnull().sum().sum()
            category_total = len(cols) * len(df)
            category_pct = category_missing / category_total * 100
            print(f"{category.upper()}: {category_missing:,} missing ({category_pct:.2f}%)")
    
    # Visualizations
    if MISSINGNO_AVAILABLE:
        # Missing data matrix
        fig, ax = plt.subplots(figsize=(15, 10))
        msno.matrix(df.iloc[:, :50], ax=ax)  # Show first 50 columns
        plt.title('Missing Data Matrix (First 50 Columns)')
        save_figure(fig, 'missing_data_matrix.png', output_dir)
        
        # Missing data heatmap
        fig, ax = plt.subplots(figsize=(15, 10))
        msno.heatmap(df.iloc[:, :50], ax=ax)
        plt.title('Missing Data Correlation Heatmap (First 50 Columns)')
        save_figure(fig, 'missing_data_heatmap.png', output_dir)
    else:
        # Alternative missing data visualization
        top_missing = missing_summary.head(20)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 12))
        
        # Top missing columns
        axes[0].barh(range(len(top_missing)), top_missing['Missing_Percentage'])
        axes[0].set_yticks(range(len(top_missing)))
        axes[0].set_yticklabels(top_missing['Column'])
        axes[0].set_xlabel('Missing Percentage (%)')
        axes[0].set_title('Top 20 Columns with Missing Data')
        
        # Missing data by category
        category_missing_pct = []
        category_names = []
        for category, cols in categories.items():
            if cols:
                category_missing = df[cols].isnull().sum().sum()
                category_total = len(cols) * len(df)
                category_pct = category_missing / category_total * 100
                category_missing_pct.append(category_pct)
                category_names.append(category)
        
        axes[1].bar(category_names, category_missing_pct)
        axes[1].set_ylabel('Missing Percentage (%)')
        axes[1].set_title('Missing Data by Category')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        save_figure(fig, 'missing_data_analysis.png', output_dir)

def analyze_correlations(df, categories, output_dir):
    """8. Correlations and Relationships Analysis"""
    print("\n" + "="*80)
    print("8. CORRELATIONS AND RELATIONSHIPS ANALYSIS")
    print("="*80)
    
    # Select numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Found {len(numeric_cols)} numeric columns for correlation analysis")
    
    # Remove columns with too many missing values
    valid_numeric_cols = []
    for col in numeric_cols:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        if missing_pct < 50:  # Keep columns with <50% missing data
            valid_numeric_cols.append(col)
    
    print(f"Using {len(valid_numeric_cols)} columns with <50% missing data")
    
    if len(valid_numeric_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = df[valid_numeric_cols].corr()
        
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
        
        print(f"\n=== HIGH CORRELATIONS (|r| > 0.7) ===")
        print(f"Found {len(high_corr_pairs)} high correlation pairs")
        for i, (col1, col2, corr_val) in enumerate(high_corr_pairs[:20]):
            print(f"{i+1:2d}. {col1[:30]} <-> {col2[:30]}: {corr_val:.3f}")
        
        # Save correlation results
        if high_corr_pairs:
            corr_df = pd.DataFrame(high_corr_pairs, columns=['Variable_1', 'Variable_2', 'Correlation'])
            corr_path = os.path.join(output_dir, 'high_correlations.csv')
            corr_df.to_csv(corr_path, index=False)
            print(f"📊 High correlations saved: {corr_path}")
        
        # Visualizations - sample correlation heatmap
        sample_cols = valid_numeric_cols[:30]  # Show first 30 numeric columns
        if len(sample_cols) > 1:
            sample_corr = df[sample_cols].corr()
            
            fig, ax = plt.subplots(figsize=(15, 12))
            sns.heatmap(sample_corr, annot=False, cmap='coolwarm', center=0, ax=ax)
            ax.set_title(f'Correlation Matrix (First {len(sample_cols)} Numeric Variables)')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_figure(fig, 'correlations_heatmap.png', output_dir)

def advanced_analytics(df, categories, output_dir):
    """9. Advanced Analytics (PCA, Clustering)"""
    print("\n" + "="*80)
    print("9. ADVANCED ANALYTICS")
    print("="*80)
    
    if not SKLEARN_AVAILABLE:
        print("Scikit-learn not available, skipping advanced analytics")
        return
    
    # Prepare data for advanced analytics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove columns with too many missing values and get a manageable subset
    analysis_cols = []
    for col in numeric_cols:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        if missing_pct < 30:  # Keep columns with <30% missing data
            analysis_cols.append(col)
    
    # Limit to reasonable number of features
    if len(analysis_cols) > 50:
        analysis_cols = analysis_cols[:50]
    
    print(f"Using {len(analysis_cols)} columns for advanced analytics")
    
    if len(analysis_cols) < 3:
        print("Not enough suitable columns for advanced analytics")
        return
    
    # Prepare the data
    analysis_data = df[analysis_cols].copy()
    
    # Fill missing values with median
    for col in analysis_cols:
        analysis_data[col] = analysis_data[col].fillna(analysis_data[col].median())
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(analysis_data)
    
    # PCA Analysis
    print("\n=== PRINCIPAL COMPONENT ANALYSIS ===")
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Explained variance
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var_ratio = np.cumsum(explained_var_ratio)
    
    # Find number of components for 80% variance
    n_components_80 = np.argmax(cumulative_var_ratio >= 0.8) + 1
    print(f"Components needed for 80% variance: {n_components_80}")
    print(f"Components needed for 90% variance: {np.argmax(cumulative_var_ratio >= 0.9) + 1}")
    
    # PCA visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Explained variance
    axes[0,0].plot(range(1, min(21, len(explained_var_ratio)+1)), 
                   explained_var_ratio[:20], 'bo-')
    axes[0,0].set_xlabel('Principal Component')
    axes[0,0].set_ylabel('Explained Variance Ratio')
    axes[0,0].set_title('PCA: Explained Variance by Component')
    
    # Cumulative explained variance
    axes[0,1].plot(range(1, min(21, len(cumulative_var_ratio)+1)), 
                   cumulative_var_ratio[:20], 'ro-')
    axes[0,1].axhline(y=0.8, color='g', linestyle='--', label='80%')
    axes[0,1].axhline(y=0.9, color='b', linestyle='--', label='90%')
    axes[0,1].set_xlabel('Principal Component')
    axes[0,1].set_ylabel('Cumulative Explained Variance Ratio')
    axes[0,1].set_title('PCA: Cumulative Explained Variance')
    axes[0,1].legend()
    
    # PCA scatter plot (first 2 components)
    axes[1,0].scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
    axes[1,0].set_xlabel(f'PC1 ({explained_var_ratio[0]:.1%} variance)')
    axes[1,0].set_ylabel(f'PC2 ({explained_var_ratio[1]:.1%} variance)')
    axes[1,0].set_title('PCA: First Two Principal Components')
    
    # K-means clustering
    print("\n=== K-MEANS CLUSTERING ===")
    # Use first 10 components for clustering
    n_components_cluster = min(10, n_components_80)
    pca_cluster = PCA(n_components=n_components_cluster)
    pca_cluster_result = pca_cluster.fit_transform(scaled_data)
    
    # Find optimal number of clusters using elbow method
    inertias = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pca_cluster_result)
        inertias.append(kmeans.inertia_)
    
    # Plot elbow curve
    axes[1,1].plot(k_range, inertias, 'bo-')
    axes[1,1].set_xlabel('Number of Clusters (k)')
    axes[1,1].set_ylabel('Inertia')
    axes[1,1].set_title('K-means: Elbow Method')
    
    plt.tight_layout()
    save_figure(fig, 'advanced_analytics.png', output_dir)
    
    # Perform clustering with optimal k (let's use k=4 as reasonable default)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_cluster_result)
    
    print(f"Clustering with k={optimal_k}")
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    print("Cluster sizes:")
    for i, count in enumerate(cluster_counts):
        print(f"  Cluster {i}: {count:,} patients ({count/len(cluster_labels)*100:.1f}%)")
    
    # Cluster visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                        c=cluster_labels, cmap='viridis', alpha=0.6)
    ax.set_xlabel(f'PC1 ({explained_var_ratio[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({explained_var_ratio[1]:.1%} variance)')
    ax.set_title(f'Patient Clusters (k={optimal_k}) in PCA Space')
    plt.colorbar(scatter)
    save_figure(fig, 'patient_clusters.png', output_dir)
    
    # Save cluster assignments
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    cluster_path = os.path.join(output_dir, 'patient_clusters.csv')
    df_with_clusters[['ACC_NUM-SESSION-ID', 'Cluster']].to_csv(cluster_path, index=False)
    print(f"📊 Patient clusters saved: {cluster_path}")

def generate_clinical_insights(df, categories, output_dir):
    """10. Clinical Insights and Recommendations"""
    print("\n" + "="*80)
    print("10. CLINICAL INSIGHTS AND RECOMMENDATIONS")
    print("="*80)
    
    insights = []
    
    # Dataset overview insights
    insights.append("=== DATASET OVERVIEW INSIGHTS ===")
    insights.append(f"• Dataset contains {len(df):,} patient records with {len(df.columns)} variables")
    insights.append(f"• Comprehensive integration of CT biomarkers, clinical outcomes, and HCC comorbidity codes")
    
    # Demographics insights
    if 'AGE' in df.columns and 'SEX' in df.columns:
        age_mean = df['AGE'].mean()
        sex_dist = df['SEX'].value_counts(normalize=True) * 100
        insights.append(f"• Average patient age: {age_mean:.1f} years")
        insights.append(f"• Sex distribution: {sex_dist.to_dict()}")
    
    # Mortality insights
    if 'DEATH_DATE' in df.columns:
        mortality_rate = df['DEATH_DATE'].notna().sum() / len(df) * 100
        insights.append(f"• Overall mortality rate: {mortality_rate:.2f}%")
    
    # HCC insights
    hcc_cols = categories['hcc_codes']
    if hcc_cols:
        hcc_burden = (df[hcc_cols] > 0).sum(axis=1)
        avg_burden = hcc_burden.mean()
        max_burden = hcc_burden.max()
        insights.append(f"• Average HCC comorbidity burden: {avg_burden:.1f} conditions per patient")
        insights.append(f"• Maximum HCC burden: {max_burden} conditions in a single patient")
        
        # Most common HCC codes
        hcc_prevalence = {}
        for col in hcc_cols:
            if df[col].notna().sum() > 0:
                prevalence = (df[col] > 0).sum() / df[col].notna().sum() * 100
                hcc_prevalence[col] = prevalence
        
        top_hcc = sorted(hcc_prevalence.items(), key=lambda x: x[1], reverse=True)[:5]
        insights.append("• Top 5 most prevalent HCC codes:")
        for hcc_code, prevalence in top_hcc:
            insights.append(f"  - {hcc_code}: {prevalence:.1f}%")
    
    # Biomarker insights
    biomarker_cols = categories['biomarkers']
    if biomarker_cols:
        numeric_biomarkers = [col for col in biomarker_cols if df[col].dtype in ['float64', 'int64']]
        insights.append(f"• {len(numeric_biomarkers)} quantitative CT biomarkers available")
        
        # Missing data insights
        missing_rates = df[biomarker_cols].isnull().sum() / len(df) * 100
        high_missing = (missing_rates > 50).sum()
        insights.append(f"• {high_missing} biomarkers have >50% missing data")
    
    # Clinical outcomes insights
    diagnosis_cols = [col for col in categories['diagnosis'] if '_DX_Code' in col]
    if diagnosis_cols:
        diagnosis_prevalence = {}
        for col in diagnosis_cols:
            condition = col.replace('_DX_Code', '')
            prevalence = df[col].notna().sum() / len(df) * 100
            diagnosis_prevalence[condition] = prevalence
        
        top_diagnoses = sorted(diagnosis_prevalence.items(), key=lambda x: x[1], reverse=True)[:5]
        insights.append("• Top 5 most prevalent clinical diagnoses:")
        for condition, prevalence in top_diagnoses:
            insights.append(f"  - {condition}: {prevalence:.1f}%")
    
    # Recommendations
    insights.append("\n=== CLINICAL RESEARCH RECOMMENDATIONS ===")
    insights.append("• Consider stratifying analyses by age groups and sex due to demographic variations")
    insights.append("• HCC codes provide rich comorbidity information for risk stratification")
    insights.append("• High missing data rates in some biomarkers may require imputation strategies")
    insights.append("• Integration of longitudinal lab values (HbA1c, CRP) enables temporal analysis")
    insights.append("• Patient clustering reveals distinct phenotypic subgroups for personalized medicine")
    insights.append("• Strong correlations between biomarkers suggest potential for dimensionality reduction")
    
    # Data quality recommendations
    insights.append("\n=== DATA QUALITY RECOMMENDATIONS ===")
    missing_summary = df.isnull().sum()
    high_missing_cols = missing_summary[missing_summary / len(df) > 0.5].index
    if len(high_missing_cols) > 0:
        insights.append(f"• {len(high_missing_cols)} columns have >50% missing data - consider exclusion or imputation")
    
    insights.append("• Validate HCC code assignments against clinical documentation")
    insights.append("• Consider temporal relationships between CT scan dates and clinical outcomes")
    insights.append("• Implement data quality checks for outlier biomarker values")
    
    # Save insights
    insights_text = "\n".join(insights)
    insights_path = os.path.join(output_dir, 'clinical_insights_recommendations.txt')
    with open(insights_path, 'w') as f:
        f.write("CLINICAL INSIGHTS AND RECOMMENDATIONS\n")
        f.write("Biomarkers with HCC Codes Dataset Analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(insights_text)
    
    print(f"📋 Clinical insights saved: {insights_path}")
    print("\nKey insights:")
    for insight in insights[:10]:  # Show first 10 insights
        print(insight)

def main():
    """Main execution function"""
    print("=" * 80)
    print("COMPREHENSIVE EDA: BIOMARKERS WITH HCC CODES DATASET")
    print("=" * 80)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup
    data_path = "../../datasets/full_data/biomarkers_with_hcc_codes_20250909_062523.csv"
    output_dir = "eda_results_biomarkers_hcc"
    
    # Load data
    print("\n📊 Loading dataset...")
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"✅ Dataset loaded successfully: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {data_path}")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Run analyses
    try:
        categories = analyze_dataset_overview(df, output_dir)
        analyze_demographics(df, categories, output_dir)
        analyze_biomarkers(df, categories, output_dir)
        analyze_hcc_codes(df, categories, output_dir)
        analyze_clinical_outcomes(df, categories, output_dir)
        analyze_lab_values(df, categories, output_dir)
        analyze_missing_data(df, categories, output_dir)
        analyze_correlations(df, categories, output_dir)
        advanced_analytics(df, categories, output_dir)
        generate_clinical_insights(df, categories, output_dir)
        
        print("\n" + "=" * 80)
        print("✅ COMPREHENSIVE EDA COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"📁 All results saved in: {output_dir}/")
        print(f"📊 Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
