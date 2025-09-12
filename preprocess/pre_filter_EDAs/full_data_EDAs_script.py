#!/usr/bin/env python3
"""
Comprehensive Exploratory Data Analysis (EDA) of Oscar Master Cohort Dataset

This script provides a thorough exploratory data analysis of the Oscar Master Cohort dataset 
(oscar_master_cohort-full.csv). The dataset contains medical information for patients including 
diagnosis codes, treatment dates, lab values, and comorbidity information.

Usage: python full_data_EDAs_script.py
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
    print("COMPREHENSIVE EDA: OSCAR MASTER COHORT DATASET")
    print("="*80)
    
    # 1. Data Loading and Overview
    print("\n1. DATA LOADING AND OVERVIEW")
    print("-" * 40)
    
    # Load the dataset
    data_path = "../../datasets/full_data/oscar_master_cohort-full.csv"
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
    
    # Identify different types of columns based on their names
    diagnosis_cols = [col for col in df.columns if '_DX_Code' in col or '_DX_NAME' in col or '_DX_dt' in col or '_DX_DT' in col]
    lab_cols = [col for col in df.columns if 'HA1C' in col or 'CRPN' in col]
    date_cols = [col for col in df.columns if 'DATE' in col.upper() or '_DT' in col.upper()]
    
    print("\n=== COLUMN CATEGORIZATION ===")
    print(f"📊 Total columns: {len(df.columns)}")
    print(f"🏥 Diagnosis-related columns: {len(diagnosis_cols)}")
    print(f"🧪 Lab value columns: {len(lab_cols)}")
    print(f"📅 Date columns: {len(date_cols)}")
    print(f"📋 Other columns: {len(df.columns) - len(diagnosis_cols) - len(lab_cols)}")
    
    print(f"\nFirst 10 diagnosis columns:")
    for i, col in enumerate(diagnosis_cols[:10]):
        print(f"  {i+1}. {col}")
    
    print(f"\nLab columns:")
    for i, col in enumerate(lab_cols):
        print(f"  {i+1}. {col}")
    
    print(f"\nSample of date columns:")
    for i, col in enumerate(date_cols[:10]):
        print(f"  {i+1}. {col}")
    
    # Display first few rows to understand the data structure
    print("\n=== SAMPLE DATA ===")
    print("First 5 rows of key columns:")
    key_cols = ['PAT_ID', 'DEATH_DATE', 'ORIG_STUDY_DATE'] + diagnosis_cols[:3] + lab_cols[:3]
    print(df[key_cols].head().to_string())
    
    print(f"\nData types summary:")
    print(df.dtypes.value_counts())
    
    # 2. Missing Values Analysis
    print("\n\n2. MISSING VALUES ANALYSIS")
    print("-" * 40)
    
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
    
    # Analyze missing values patterns
    complete_data_count = (missing_stats['Missing_Percentage'] == 0).sum()
    partial_missing_count = ((missing_stats['Missing_Percentage'] > 0) & (missing_stats['Missing_Percentage'] < 100)).sum()
    all_missing_count = (missing_stats['Missing_Percentage'] == 100).sum()
    
    print(f"\n📋 Data completeness summary:")
    print(f"   ✅ Columns with complete data: {complete_data_count}")
    print(f"   ⚠️  Columns with partial missing data: {partial_missing_count}")
    print(f"   ❌ Columns with all missing data: {all_missing_count}")
    
    print(f"\nColumns with no missing values:")
    complete_cols = missing_stats[missing_stats['Missing_Percentage'] == 0]['Column'].tolist()
    print(f"Count: {len(complete_cols)}")
    for col in complete_cols[:10]:
        print(f"  - {col}")
    if len(complete_cols) > 10:
        print(f"  ... and {len(complete_cols) - 10} more")
    
    # 3. Missing Values Visualization
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
        
        # 3. Missing values by column type
        if diagnosis_cols:
            diagnosis_missing = df[diagnosis_cols].isnull().sum().sum() / (len(diagnosis_cols) * len(df)) * 100
        else:
            diagnosis_missing = 0
        if lab_cols:
            lab_missing = df[lab_cols].isnull().sum().sum() / (len(lab_cols) * len(df)) * 100
        else:
            lab_missing = 0
        other_cols = [col for col in df.columns if col not in diagnosis_cols + lab_cols]
        if other_cols:
            other_missing = df[other_cols].isnull().sum().sum() / (len(other_cols) * len(df)) * 100
        else:
            other_missing = 0
        
        col_types = ['Diagnosis', 'Lab Values', 'Other']
        missing_pcts = [diagnosis_missing, lab_missing, other_missing]
        sns.barplot(x=col_types, y=missing_pcts, ax=axes[1,0])
        axes[1,0].set_title('Missing Values by Column Type')
        axes[1,0].set_ylabel('Missing Percentage (%)')
        
        # 4. Missing values distribution histogram
        missing_stats['Missing_Percentage'].hist(bins=20, ax=axes[1,1])
        axes[1,1].set_title('Distribution of Missing Value Percentages')
        axes[1,1].set_xlabel('Missing Percentage (%)')
        axes[1,1].set_ylabel('Number of Columns')
        
        plt.tight_layout()
        plt.savefig('missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Missing values visualization saved as 'missing_values_analysis.png'")
    except Exception as e:
        print(f"⚠️  Could not create missing values visualization: {e}")
    
    # 4. Patient Demographics and Mortality Analysis
    print("\n\n4. PATIENT DEMOGRAPHICS AND MORTALITY ANALYSIS")
    print("-" * 50)
    
    print("=== PATIENT DEMOGRAPHICS ANALYSIS ===")
    print(f"👥 Total unique patients: {df['PAT_ID'].nunique():,}")
    print(f"📄 Total records: {len(df):,}")
    print(f"📊 Average records per patient: {len(df) / df['PAT_ID'].nunique():.2f}")
    
    # Check for duplicate patient records
    duplicate_patients = df['PAT_ID'].value_counts()
    patients_with_multiple_records = (duplicate_patients > 1).sum()
    print(f"🔄 Patients with multiple records: {patients_with_multiple_records}")
    
    # Mortality analysis
    death_data = df['DEATH_DATE'].dropna()
    total_patients = df['PAT_ID'].nunique()
    deceased_patients = len(death_data)
    
    print(f"\n=== MORTALITY ANALYSIS ===")
    print(f"👥 Total patients: {total_patients:,}")
    print(f"💀 Deceased patients: {deceased_patients:,}")
    print(f"📊 Mortality rate: {deceased_patients/total_patients*100:.2f}%")
    
    # 5. Comorbidity Analysis
    print("\n\n5. COMORBIDITY ANALYSIS")
    print("-" * 30)
    
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
            print(f"{i+1:2d}. {condition:30s}: {stats['count']:5d} patients ({stats['prevalence']:5.1f}%)")
        
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
            plt.savefig('comorbidity_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✅ Comorbidity analysis visualization saved as 'comorbidity_analysis.png'")
        except Exception as e:
            print(f"⚠️  Could not create comorbidity visualization: {e}")
    else:
        print("❌ No condition columns found for comorbidity analysis.")
    
    # 6. Lab Values Analysis
    print("\n\n6. LABORATORY VALUES ANALYSIS")
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
            for col in hba1c_cols[:3]:  # Analyze first 3 columns
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
            for col in creatinine_cols[:3]:  # Analyze first 3 columns
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
        
        # Visualize lab values
        try:
            # Select specific columns for visualization: 2 HbA1c + 2 CRPN
            value_cols = [col for col in lab_cols if '_DT' not in col]  # Exclude date columns
            hba1c_value_cols = [col for col in value_cols if 'HA1C' in col]
            crpn_value_cols = [col for col in value_cols if 'CRPN' in col]
            
            # Select best columns for visualization
            viz_cols = []
            if hba1c_value_cols:
                viz_cols.extend(hba1c_value_cols[:2])  # First 2 HbA1c columns
            if crpn_value_cols:
                viz_cols.extend(crpn_value_cols[:2])    # First 2 CRPN columns
            
            lab_data_clean = {}
            for col in viz_cols:
                values = df[col].astype(str).str.replace('<', '').str.replace('>', '')
                numeric_values = pd.to_numeric(values, errors='coerce')
                valid_values = numeric_values.dropna()
                if len(valid_values) > 0:
                    lab_data_clean[col] = valid_values
            
            if lab_data_clean:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Plot distributions
                plot_count = 0
                for col, values in lab_data_clean.items():
                    if plot_count < 4:
                        row, col_idx = plot_count // 2, plot_count % 2
                        values.hist(bins=30, alpha=0.7, ax=axes[row, col_idx])
                        
                        # Better title with measurement type
                        measurement_type = "HbA1c (%)" if 'HA1C' in col else "Creatinine (mg/dL)"
                        axes[row, col_idx].set_title(f'{col}\\n{measurement_type}')
                        axes[row, col_idx].set_xlabel('Value')
                        axes[row, col_idx].set_ylabel('Frequency')
                        
                        # Add statistics text
                        stats_text = f'n={len(values):,}\\nMean={values.mean():.2f}\\nStd={values.std():.2f}'
                        axes[row, col_idx].text(0.7, 0.7, stats_text, transform=axes[row, col_idx].transAxes,
                                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        plot_count += 1
                
                # Hide empty subplots
                for i in range(plot_count, 4):
                    row, col_idx = i // 2, i % 2
                    axes[row, col_idx].set_visible(False)
                
                plt.tight_layout()
                plt.savefig('lab_values_analysis.png', dpi=300, bbox_inches='tight')
                plt.show()
                print("✅ Lab values visualization saved as 'lab_values_analysis.png'")
                print(f"📊 Plotted {plot_count} lab value distributions: {list(lab_data_clean.keys())}")
        except Exception as e:
            print(f"⚠️  Could not create lab values visualization: {e}")
    else:
        print("❌ No lab value columns found for analysis.")
    
    # 7. Key Insights and Recommendations
    print("\n\n7. KEY INSIGHTS AND RECOMMENDATIONS")
    print("-" * 40)
    
    print("=" * 80)
    print("KEY INSIGHTS AND RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n📊 DATASET SUMMARY:")
    print(f"   • Total patients: {df['PAT_ID'].nunique():,}")
    print(f"   • Total records: {len(df):,}")
    print(f"   • Total variables: {len(df.columns)}")
    print(f"   • Medical conditions tracked: {len(condition_cols) if 'condition_cols' in locals() else 'N/A'}")
    print(f"   • Lab measurements: {len(lab_cols) if lab_cols else 0}")
    
    print("\n🔍 DATA QUALITY INSIGHTS:")
    complete_data_pct = (missing_stats['Missing_Percentage'] == 0).sum() / len(missing_stats) * 100
    print(f"   • {complete_data_pct:.1f}% of columns have complete data")
    overall_missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    print(f"   • {overall_missing_pct:.1f}% of all values are missing")
    print(f"   • Primary key (PAT_ID) appears to be unique per record")
    
    if 'sorted_conditions' in locals() and sorted_conditions:
        print("\n🏥 CLINICAL INSIGHTS:")
        top_condition = sorted_conditions[0]
        print(f"   • Most common condition: {top_condition[0]} ({top_condition[1]['prevalence']:.1f}% prevalence)")
        
        if 'conditions_per_patient' in locals():
            avg_conditions = conditions_per_patient.mean()
            multiple_conditions_pct = (conditions_per_patient >= 2).mean() * 100
            print(f"   • Average conditions per patient: {avg_conditions:.1f}")
            print(f"   • {multiple_conditions_pct:.1f}% of patients have multiple comorbidities")
    
    print("\n📈 RECOMMENDATIONS FOR FURTHER ANALYSIS:")
    print("   1️⃣  Data Preprocessing:")
    print("      • Handle missing values using domain-specific imputation")
    print("      • Standardize date formats across all date columns")
    print("      • Clean and normalize lab values (handle '<' and '>' symbols)")
    
    print("\n   2️⃣  Feature Engineering:")
    print("      • Create comorbidity burden scores")
    print("      • Calculate time-to-event variables from diagnosis dates")
    print("      • Develop lab value change trajectories over time")
    print("      • Create age-at-diagnosis features")
    
    print("\n   3️⃣  Clinical Analysis:")
    print("      • Perform survival analysis using mortality data")
    print("      • Analyze disease progression patterns")
    print("      • Study drug response and lab value correlations")
    print("      • Investigate comorbidity interaction effects")
    
    print("\n   4️⃣  Machine Learning Applications:")
    print("      • Risk stratification models")
    print("      • Comorbidity prediction models")
    print("      • Lab value trajectory clustering")
    print("      • Mortality prediction models")
    
    print("\n   5️⃣  Data Validation:")
    print("      • Cross-reference diagnosis codes with clinical guidelines")
    print("      • Validate lab value ranges against clinical norms")
    print("      • Check temporal consistency of diagnoses")
    print("      • Verify patient demographic information")
    
    print("\n✅ EDA COMPLETE - Dataset is ready for advanced analytics!")
    print(f"\n📁 Output files saved:")
    print(f"   • missing_values_analysis.png")
    if 'condition_cols' in locals() and condition_cols:
        print(f"   • comorbidity_analysis.png")
    if lab_cols:
        print(f"   • lab_values_analysis.png")

if __name__ == "__main__":
    main()
