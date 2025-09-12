#!/usr/bin/env python3
"""
Comprehensive Diagnostic Names Analysis for Oscar Master Cohort Dataset

This script provides detailed exploratory data analysis of diagnostic names
for the top 15 most prevalent and 5 rarest medical conditions.

Usage: python diagnostic_names_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

def analyze_diagnostic_names():
    """Main function to analyze diagnostic names comprehensively"""
    
    print("="*80)
    print("COMPREHENSIVE DIAGNOSTIC NAMES ANALYSIS")
    print("Oscar Master Cohort Dataset")
    print("="*80)
    
    # Load the dataset
    df = pd.read_csv('../../datasets/full_data/oscar_master_cohort-full.csv', low_memory=False)
    
    # Get condition prevalence
    condition_cols = [col for col in df.columns if '_DX_Code' in col and col.replace('_DX_Code', '') + '_DX_NAME' in df.columns]
    
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
    
    # Define condition sets for analysis
    top_15 = [c[0] for c in sorted_conditions[:15]]
    rare_conditions = [(c, s) for c, s in sorted_conditions if s['count'] > 0][-5:]
    rarest_5 = [c[0] for c in rare_conditions]
    
    print(f"\n📊 ANALYSIS SCOPE:")
    print(f"   • Top 15 most prevalent conditions")
    print(f"   • Top 5 rarest conditions (with >0 patients)")
    print(f"   • Total conditions analyzed: {len(top_15) + len(rarest_5)}")
    
    # Analyze top 15 conditions
    print(f"\n\n1. TOP 15 MOST PREVALENT CONDITIONS - DIAGNOSTIC NAMES ANALYSIS")
    print("="*75)
    
    analyze_condition_group(df, top_15, "Top_15_Prevalent", "Most Prevalent")
    
    # Analyze rarest 5 conditions
    print(f"\n\n2. TOP 5 RAREST CONDITIONS - DIAGNOSTIC NAMES ANALYSIS")
    print("="*75)
    
    analyze_condition_group(df, rarest_5, "Rarest_5", "Rarest")
    
    # Comparative analysis
    print(f"\n\n3. COMPARATIVE ANALYSIS: PREVALENT vs RARE CONDITIONS")
    print("="*75)
    
    comparative_analysis(df, top_15, rarest_5)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Generated visualizations:")
    print(f"   • diagnostic_names_top15_analysis.png")
    print(f"   • diagnostic_names_rare_analysis.png") 
    print(f"   • diagnostic_names_comparative.png")
    print(f"   • diagnostic_wordclouds.png")

def analyze_condition_group(df, conditions, group_name, group_description):
    """Analyze diagnostic names for a group of conditions"""
    
    all_diagnostic_data = []
    
    for condition in conditions:
        name_col = f'{condition}_DX_NAME'
        code_col = f'{condition}_DX_Code'
        
        if name_col in df.columns:
            # Get diagnostic names for this condition
            condition_data = df[df[name_col].notna()]
            diagnostic_names = condition_data[name_col].tolist()
            diagnostic_codes = condition_data[code_col].tolist()
            
            print(f"\n📋 {condition.replace('_', ' ').title()}:")
            print(f"   Patients: {len(diagnostic_names):,}")
            
            # Analyze diagnostic name diversity
            unique_names = list(set(diagnostic_names))
            unique_codes = list(set(diagnostic_codes))
            
            print(f"   Unique diagnostic descriptions: {len(unique_names)}")
            print(f"   Unique diagnostic codes: {len(unique_codes)}")
            print(f"   Avg descriptions per code: {len(unique_names)/len(unique_codes):.1f}")
            
            # Most common diagnostic names
            name_counts = Counter(diagnostic_names)
            print(f"   Top 3 diagnostic descriptions:")
            for i, (name, count) in enumerate(name_counts.most_common(3)):
                pct = count/len(diagnostic_names)*100
                print(f"     {i+1}. {name[:60]}{'...' if len(name) > 60 else ''} ({count:,} patients, {pct:.1f}%)")
            
            # Analyze text characteristics
            text_analysis = analyze_diagnostic_text(diagnostic_names)
            print(f"   Text characteristics:")
            print(f"     Avg words per description: {text_analysis['avg_words']:.1f}")
            print(f"     Avg characters per description: {text_analysis['avg_chars']:.1f}")
            print(f"     Contains 'HCC': {text_analysis['hcc_count']:,} descriptions ({text_analysis['hcc_pct']:.1f}%)")
            
            # Store data for visualization
            all_diagnostic_data.append({
                'condition': condition,
                'names': diagnostic_names,
                'unique_names': unique_names,
                'name_counts': name_counts,
                'text_analysis': text_analysis
            })
    
    # Create visualizations for this group
    create_group_visualizations(all_diagnostic_data, group_name, group_description)

def analyze_diagnostic_text(diagnostic_names):
    """Analyze text characteristics of diagnostic names"""
    
    word_counts = []
    char_counts = []
    hcc_count = 0
    
    for name in diagnostic_names:
        words = len(str(name).split())
        chars = len(str(name))
        word_counts.append(words)
        char_counts.append(chars)
        
        if 'HCC' in str(name).upper():
            hcc_count += 1
    
    return {
        'avg_words': np.mean(word_counts),
        'avg_chars': np.mean(char_counts),
        'hcc_count': hcc_count,
        'hcc_pct': hcc_count/len(diagnostic_names)*100 if diagnostic_names else 0
    }

def create_group_visualizations(data, group_name, group_description):
    """Create comprehensive visualizations for a condition group"""
    
    # Set up the plot
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Diagnostic diversity (unique descriptions per condition)
    ax1 = plt.subplot(3, 3, 1)
    conditions = [d['condition'].replace('_', ' ').title() for d in data]
    unique_counts = [len(d['unique_names']) for d in data]
    
    bars = ax1.bar(range(len(conditions)), unique_counts, alpha=0.7)
    ax1.set_title(f'{group_description} Conditions\nDiagnostic Description Diversity')
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Unique Descriptions')
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], 
                        rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, unique_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                str(count), ha='center', va='bottom', fontsize=8)
    
    # 2. Patient volume per condition
    ax2 = plt.subplot(3, 3, 2)
    patient_counts = [len(d['names']) for d in data]
    
    bars = ax2.bar(range(len(conditions)), patient_counts, alpha=0.7, color='orange')
    ax2.set_title(f'{group_description} Conditions\nPatient Volume')
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('Number of Patients')
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], 
                        rotation=45, ha='right')
    
    # Add value labels
    for bar, count in zip(bars, patient_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(patient_counts)*0.01, 
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    # 3. Description complexity (average words per description)
    ax3 = plt.subplot(3, 3, 3)
    avg_words = [d['text_analysis']['avg_words'] for d in data]
    
    bars = ax3.bar(range(len(conditions)), avg_words, alpha=0.7, color='green')
    ax3.set_title(f'{group_description} Conditions\nDescription Complexity')
    ax3.set_xlabel('Condition')
    ax3.set_ylabel('Avg Words per Description')
    ax3.set_xticks(range(len(conditions)))
    ax3.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], 
                        rotation=45, ha='right')
    
    # 4. HCC prevalence in descriptions
    ax4 = plt.subplot(3, 3, 4)
    hcc_percentages = [d['text_analysis']['hcc_pct'] for d in data]
    
    bars = ax4.bar(range(len(conditions)), hcc_percentages, alpha=0.7, color='red')
    ax4.set_title(f'{group_description} Conditions\nHCC Designation Prevalence')
    ax4.set_xlabel('Condition')
    ax4.set_ylabel('% Descriptions with HCC')
    ax4.set_xticks(range(len(conditions)))
    ax4.set_xticklabels([c[:15] + '...' if len(c) > 15 else c for c in conditions], 
                        rotation=45, ha='right')
    
    # 5. Top diagnostic descriptions (word cloud for top 3 conditions)
    if len(data) >= 3:
        ax5 = plt.subplot(3, 3, 5)
        # Combine top diagnostic names from top 3 conditions
        top_3_names = []
        for i in range(min(3, len(data))):
            # Get top 5 descriptions for each condition
            top_names = [name for name, count in data[i]['name_counts'].most_common(5)]
            top_3_names.extend(top_names)
        
        if top_3_names:
            try:
                # Create word cloud
                text = ' '.join(top_3_names)
                # Clean text for word cloud
                text = re.sub(r'[^\w\s]', ' ', text)
                
                wordcloud = WordCloud(width=400, height=300, background_color='white',
                                    max_words=50, colormap='viridis').generate(text)
                ax5.imshow(wordcloud, interpolation='bilinear')
                ax5.axis('off')
                ax5.set_title(f'Common Terms in\\n{group_description} Diagnoses')
            except Exception as e:
                ax5.text(0.5, 0.5, f'Word cloud\\nnot available\\n({str(e)[:20]})', 
                        ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Diagnostic Terms')
    
    # 6. Description length distribution
    ax6 = plt.subplot(3, 3, 6)
    all_char_lengths = []
    for d in data:
        char_lengths = [len(str(name)) for name in d['names']]
        all_char_lengths.extend(char_lengths)
    
    if all_char_lengths:
        ax6.hist(all_char_lengths, bins=30, alpha=0.7, edgecolor='black')
        ax6.set_title(f'{group_description} Conditions\nDescription Length Distribution')
        ax6.set_xlabel('Characters per Description')
        ax6.set_ylabel('Frequency')
        ax6.axvline(np.mean(all_char_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_char_lengths):.1f}')
        ax6.legend()
    
    # 7. Diagnostic diversity vs patient volume scatter
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(patient_counts, unique_counts, alpha=0.7, s=100)
    ax7.set_xlabel('Number of Patients')
    ax7.set_ylabel('Unique Descriptions')
    ax7.set_title(f'{group_description} Conditions\nDiversity vs Volume')
    
    # Add condition labels
    for i, condition in enumerate(conditions):
        ax7.annotate(condition[:10], (patient_counts[i], unique_counts[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 8. Most common diagnostic terms (bar chart)
    ax8 = plt.subplot(3, 3, 8)
    all_words = []
    for d in data:
        for name in d['unique_names']:
            # Extract meaningful words (skip common medical terms)
            words = re.findall(r'\b[a-zA-Z]{4,}\b', str(name).lower())
            all_words.extend(words)
    
    if all_words:
        # Remove common stop words
        stop_words = {'unspecified', 'without', 'mention', 'disease', 'with', 'type', 'other', 
                     'chronic', 'acute', 'primary', 'secondary', 'disorder', 'syndrome'}
        filtered_words = [w for w in all_words if w not in stop_words]
        
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(10)
        
        if top_words:
            words, counts = zip(*top_words)
            bars = ax8.barh(range(len(words)), counts, alpha=0.7)
            ax8.set_yticks(range(len(words)))
            ax8.set_yticklabels(words)
            ax8.set_xlabel('Frequency')
            ax8.set_title(f'{group_description} Conditions\nMost Common Terms')
            ax8.invert_yaxis()
    
    # 9. Summary statistics table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Create summary table
    summary_data = []
    for d in data:
        condition_name = d['condition'].replace('_', ' ').title()
        summary_data.append([
            condition_name[:15] + '...' if len(condition_name) > 15 else condition_name,
            f"{len(d['names']):,}",
            f"{len(d['unique_names'])}",
            f"{d['text_analysis']['avg_words']:.1f}",
            f"{d['text_analysis']['hcc_pct']:.1f}%"
        ])
    
    # Create table
    table = ax9.table(cellText=summary_data,
                     colLabels=['Condition', 'Patients', 'Unique Desc', 'Avg Words', 'HCC %'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax9.set_title(f'{group_description} Conditions Summary')
    
    plt.tight_layout()
    plt.savefig(f'diagnostic_names_{group_name.lower()}_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Visualization saved: diagnostic_names_{group_name.lower()}_analysis.png")

def comparative_analysis(df, top_15, rarest_5):
    """Compare diagnostic characteristics between prevalent and rare conditions"""
    
    # Analyze both groups
    prevalent_data = get_group_statistics(df, top_15)
    rare_data = get_group_statistics(df, rarest_5)
    
    print(f"\n📊 COMPARATIVE STATISTICS:")
    print(f"{'Metric':<35} {'Prevalent':<15} {'Rare':<15} {'Difference'}")
    print("-" * 70)
    
    metrics = [
        ('Avg patients per condition', 'avg_patients'),
        ('Avg unique descriptions', 'avg_unique_desc'),
        ('Avg words per description', 'avg_words'),
        ('Avg chars per description', 'avg_chars'),
        ('% with HCC designation', 'avg_hcc_pct'),
        ('Description diversity ratio', 'diversity_ratio')
    ]
    
    for metric_name, key in metrics:
        prev_val = prevalent_data[key]
        rare_val = rare_data[key]
        diff = prev_val - rare_val
        
        if 'pct' in key or 'ratio' in key:
            print(f"{metric_name:<35} {prev_val:<15.1f} {rare_val:<15.1f} {diff:+.1f}")
        else:
            print(f"{metric_name:<35} {prev_val:<15.0f} {rare_val:<15.0f} {diff:+.0f}")
    
    # Create comparative visualization
    create_comparative_visualization(prevalent_data, rare_data)
    
    print(f"\n📋 KEY INSIGHTS:")
    print(f"   • Prevalent conditions have {prevalent_data['avg_patients']/rare_data['avg_patients']:.1f}x more patients on average")
    print(f"   • Rare conditions have {rare_data['avg_unique_desc']/prevalent_data['avg_unique_desc']:.1f}x more diagnostic diversity")
    print(f"   • HCC designation is {prevalent_data['avg_hcc_pct']/rare_data['avg_hcc_pct']:.1f}x more common in prevalent conditions")

def get_group_statistics(df, conditions):
    """Calculate aggregate statistics for a group of conditions"""
    
    total_patients = 0
    total_unique_desc = 0
    all_text_metrics = []
    all_hcc_pcts = []
    
    for condition in conditions:
        name_col = f'{condition}_DX_NAME'
        
        if name_col in df.columns:
            diagnostic_names = df[df[name_col].notna()][name_col].tolist()
            unique_names = list(set(diagnostic_names))
            text_analysis = analyze_diagnostic_text(diagnostic_names)
            
            total_patients += len(diagnostic_names)
            total_unique_desc += len(unique_names)
            all_text_metrics.append(text_analysis)
            all_hcc_pcts.append(text_analysis['hcc_pct'])
    
    return {
        'avg_patients': total_patients / len(conditions),
        'avg_unique_desc': total_unique_desc / len(conditions),
        'avg_words': np.mean([m['avg_words'] for m in all_text_metrics]),
        'avg_chars': np.mean([m['avg_chars'] for m in all_text_metrics]),
        'avg_hcc_pct': np.mean(all_hcc_pcts),
        'diversity_ratio': total_unique_desc / total_patients * 100
    }

def create_comparative_visualization(prevalent_data, rare_data):
    """Create comparative visualization between prevalent and rare conditions"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Data for comparison
    metrics = ['avg_patients', 'avg_unique_desc', 'avg_words', 'avg_chars', 'avg_hcc_pct', 'diversity_ratio']
    metric_labels = ['Avg Patients', 'Avg Unique Desc', 'Avg Words', 'Avg Characters', 'HCC %', 'Diversity Ratio']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        groups = ['Prevalent', 'Rare']
        values = [prevalent_data[metric], rare_data[metric]]
        
        bars = ax.bar(groups, values, alpha=0.7, color=['blue', 'red'])
        ax.set_title(f'{label}\nComparison')
        ax.set_ylabel(label)
        
        # Add value labels
        for bar, value in zip(bars, values):
            if metric in ['avg_hcc_pct', 'diversity_ratio']:
                label_text = f'{value:.1f}'
            else:
                label_text = f'{value:.0f}'
            
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                   label_text, ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('diagnostic_names_comparative.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Comparative visualization saved: diagnostic_names_comparative.png")

if __name__ == "__main__":
    try:
        from wordcloud import WordCloud
        WORDCLOUD_AVAILABLE = True
    except ImportError:
        print("⚠️  WordCloud not available. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "wordcloud"])
        from wordcloud import WordCloud
        WORDCLOUD_AVAILABLE = True
    
    analyze_diagnostic_names()
