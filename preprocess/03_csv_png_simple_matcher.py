#!/usr/bin/env python3
"""
CSV-PNG Simple Matcher
======================

Simple and efficient matching between CSV rows and standardized PNG files.
Since both CSV ACC_NUM-SESSION-ID and PNG filenames now follow NNNNNN-NNNN format,
we can perform direct string matching.

Author: AI Assistant
Date: 2025-09-04
"""

import os
import pandas as pd
from datetime import datetime
import json


class SimpleCSVPNGMatcher:
    def __init__(self, csv_path, png_dir, output_dir="matching_results"):
        """
        Initialize the simple matcher.
        
        Args:
            csv_path: Path to the CSV file
            png_dir: Directory containing standardized PNG files
            output_dir: Directory to save matching results
        """
        self.csv_path = csv_path
        self.png_dir = png_dir
        self.output_dir = os.path.join(os.path.dirname(__file__), output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data containers
        self.csv_data = None
        self.png_files = set()
        self.csv_identifiers = set()
        
        # Matching results
        self.matched_pairs = {}  # {acc_id: png_filename}
        self.unmatched_csv = []  # ACC_NUM-SESSION-ID values with no PNG
        self.unmatched_png = []  # PNG files with no CSV match
        
        # Statistics
        self.stats = {}

    def load_data(self):
        """Load CSV data and scan PNG files."""
        print("🚀 SIMPLE CSV-PNG MATCHING")
        print("=" * 60)
        
        # Load CSV
        print("📊 Loading CSV data...")
        try:
            self.csv_data = pd.read_csv(self.csv_path, low_memory=False)
            print(f"✅ Loaded CSV: {len(self.csv_data):,} rows, {len(self.csv_data.columns)} columns")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False
        
        # Extract unique ACC_NUM-SESSION-ID values
        acc_col = 'ACC_NUM-SESSION-ID'
        if acc_col not in self.csv_data.columns:
            print(f"❌ Error: Column '{acc_col}' not found in CSV")
            return False
        
        self.csv_identifiers = set(self.csv_data[acc_col].dropna().astype(str))
        print(f"📋 Found {len(self.csv_identifiers):,} unique ACC_NUM-SESSION-ID values")
        
        # Scan PNG files
        print("🖼️  Scanning PNG files...")
        try:
            all_files = os.listdir(self.png_dir)
            png_files_list = [f for f in all_files if f.lower().endswith('.png')]
            
            # Extract identifiers from PNG filenames (remove .png extension)
            png_identifiers = [f[:-4] for f in png_files_list]  # Remove .png
            self.png_files = set(png_identifiers)
            
            print(f"✅ Found {len(png_files_list):,} PNG files")
            print(f"📋 Extracted {len(self.png_files):,} unique PNG identifiers")
            
        except Exception as e:
            print(f"❌ Error scanning PNG files: {e}")
            return False
        
        return True

    def perform_matching(self):
        """Perform direct string matching between CSV and PNG identifiers."""
        print("\n🔗 Performing direct matching...")
        
        # Find matches using set intersection
        matches = self.csv_identifiers.intersection(self.png_files)
        print(f"✅ Found {len(matches):,} direct matches")
        
        # Store matched pairs
        for identifier in matches:
            self.matched_pairs[identifier] = f"{identifier}.png"
        
        # Find unmatched CSV identifiers
        self.unmatched_csv = list(self.csv_identifiers - matches)
        print(f"📋 Unmatched CSV identifiers: {len(self.unmatched_csv):,}")
        
        # Find unmatched PNG files
        self.unmatched_png = [f"{identifier}.png" for identifier in (self.png_files - matches)]
        print(f"🖼️  Unmatched PNG files: {len(self.unmatched_png):,}")

    def calculate_statistics(self):
        """Calculate detailed matching statistics."""
        print("\n📊 Calculating statistics...")
        
        total_csv = len(self.csv_identifiers)
        total_png = len(self.png_files)
        total_matches = len(self.matched_pairs)
        
        self.stats = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'csv_file': os.path.basename(self.csv_path),
            'png_directory': os.path.basename(self.png_dir),
            'total_csv_identifiers': total_csv,
            'total_png_files': total_png,
            'successful_matches': total_matches,
            'unmatched_csv_count': len(self.unmatched_csv),
            'unmatched_png_count': len(self.unmatched_png),
            'csv_match_rate': (total_matches / total_csv * 100) if total_csv > 0 else 0,
            'png_match_rate': (total_matches / total_png * 100) if total_png > 0 else 0,
            'overall_efficiency': (total_matches / max(total_csv, total_png) * 100) if max(total_csv, total_png) > 0 else 0
        }
        
        print(f"📈 CSV Match Rate: {self.stats['csv_match_rate']:.2f}%")
        print(f"📈 PNG Match Rate: {self.stats['png_match_rate']:.2f}%")
        print(f"📈 Overall Efficiency: {self.stats['overall_efficiency']:.2f}%")

    def save_results(self):
        """Save matching results to files."""
        print("\n💾 Saving results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save matched pairs
        if self.matched_pairs:
            matched_df = pd.DataFrame([
                {'acc_num_session_id': acc_id, 'png_filename': png_file}
                for acc_id, png_file in self.matched_pairs.items()
            ])
            matched_file = os.path.join(self.output_dir, f"matched_pairs_{timestamp}.csv")
            matched_df.to_csv(matched_file, index=False)
            print(f"✅ Matched pairs saved: {matched_file}")
        
        # Save unmatched CSV identifiers
        if self.unmatched_csv:
            unmatched_csv_df = pd.DataFrame({
                'unmatched_acc_num_session_id': self.unmatched_csv
            })
            unmatched_csv_file = os.path.join(self.output_dir, f"unmatched_csv_{timestamp}.csv")
            unmatched_csv_df.to_csv(unmatched_csv_file, index=False)
            print(f"📋 Unmatched CSV saved: {unmatched_csv_file}")
        
        # Save unmatched PNG files
        if self.unmatched_png:
            unmatched_png_df = pd.DataFrame({
                'unmatched_png_filename': self.unmatched_png
            })
            unmatched_png_file = os.path.join(self.output_dir, f"unmatched_png_{timestamp}.csv")
            unmatched_png_df.to_csv(unmatched_png_file, index=False)
            print(f"🖼️  Unmatched PNG saved: {unmatched_png_file}")
        
        # Save statistics as JSON
        stats_file = os.path.join(self.output_dir, f"matching_statistics_{timestamp}.json")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"📊 Statistics saved: {stats_file}")

    def generate_report(self):
        """Generate a comprehensive text report."""
        print("\n📄 Generating comprehensive report...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"matching_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write("CSV-PNG MATCHING REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {self.stats['timestamp']}\n")
            f.write(f"CSV File: {self.stats['csv_file']}\n")
            f.write(f"PNG Directory: {self.stats['png_directory']}\n\n")
            
            f.write("MATCHING SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total CSV identifiers: {self.stats['total_csv_identifiers']:,}\n")
            f.write(f"Total PNG files: {self.stats['total_png_files']:,}\n")
            f.write(f"Successful matches: {self.stats['successful_matches']:,}\n")
            f.write(f"Unmatched CSV: {self.stats['unmatched_csv_count']:,}\n")
            f.write(f"Unmatched PNG: {self.stats['unmatched_png_count']:,}\n\n")
            
            f.write("MATCHING RATES\n")
            f.write("-" * 30 + "\n")
            f.write(f"CSV Match Rate: {self.stats['csv_match_rate']:.2f}%\n")
            f.write(f"PNG Match Rate: {self.stats['png_match_rate']:.2f}%\n")
            f.write(f"Overall Efficiency: {self.stats['overall_efficiency']:.2f}%\n\n")
            
            if self.unmatched_csv:
                f.write("SAMPLE UNMATCHED CSV IDENTIFIERS\n")
                f.write("-" * 40 + "\n")
                for i, acc_id in enumerate(self.unmatched_csv[:20]):
                    f.write(f"  {acc_id}\n")
                if len(self.unmatched_csv) > 20:
                    f.write(f"  ... and {len(self.unmatched_csv) - 20:,} more\n")
                f.write("\n")
            
            if self.unmatched_png:
                f.write("SAMPLE UNMATCHED PNG FILES\n")
                f.write("-" * 35 + "\n")
                for i, png_file in enumerate(self.unmatched_png[:20]):
                    f.write(f"  {png_file}\n")
                if len(self.unmatched_png) > 20:
                    f.write(f"  ... and {len(self.unmatched_png) - 20:,} more\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            if self.stats['csv_match_rate'] > 95:
                f.write("✅ Excellent matching rate! The standardization was very successful.\n")
            elif self.stats['csv_match_rate'] > 80:
                f.write("✅ Good matching rate. Minor data inconsistencies may exist.\n")
            else:
                f.write("⚠️  Lower matching rate. Consider investigating data quality issues.\n")
            
            f.write("1. Review unmatched CSV identifiers for format inconsistencies\n")
            f.write("2. Check unmatched PNG files for naming convention violations\n")
            f.write("3. Consider fuzzy matching for remaining unmatched pairs\n")
        
        print(f"📄 Report saved: {report_file}")

    def display_summary(self):
        """Display a summary of results to console."""
        print("\n" + "=" * 60)
        print("🎯 MATCHING SUMMARY")
        print("=" * 60)
        print(f"📊 Total CSV Records: {self.stats['total_csv_identifiers']:,}")
        print(f"🖼️  Total PNG Files: {self.stats['total_png_files']:,}")
        print(f"✅ Successful Matches: {self.stats['successful_matches']:,}")
        print(f"❌ Unmatched CSV: {self.stats['unmatched_csv_count']:,}")
        print(f"❌ Unmatched PNG: {self.stats['unmatched_png_count']:,}")
        print()
        print(f"📈 CSV Match Rate: {self.stats['csv_match_rate']:.2f}%")
        print(f"📈 PNG Match Rate: {self.stats['png_match_rate']:.2f}%")
        print(f"📈 Overall Efficiency: {self.stats['overall_efficiency']:.2f}%")
        print("=" * 60)

    def run(self):
        """Execute the complete matching process."""
        if not self.load_data():
            return False
        
        self.perform_matching()
        self.calculate_statistics()
        self.save_results()
        self.generate_report()
        self.display_summary()
        
        print(f"\n🎉 SUCCESS: Matching complete!")
        print(f"📁 Results saved to: {self.output_dir}")
        return True


def main():
    """Main execution function."""
    # Define paths
    csv_file = os.path.join(os.path.dirname(__file__), '../../datasets/full_data/2025_08_31_Biomarkers_Outcomes_Joined_Fixed.csv')
    png_directory = os.path.join(os.path.dirname(__file__), '../../datasets/full_data/data')
    
    # Create and run matcher
    matcher = SimpleCSVPNGMatcher(csv_file, png_directory)
    success = matcher.run()
    
    if success:
        print("\n✅ All operations completed successfully!")
    else:
        print("\n❌ Some operations failed. Check the output above.")


if __name__ == "__main__":
    main()
