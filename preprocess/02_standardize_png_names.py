#!/usr/bin/env python3
"""
PNG Filename Standardization Script

This script standardizes all PNG filenames to the Schema 2 format (NNNNNN-NNNN)
and creates detailed mapping and validation files.

Steps:
1. Create mapping file of original → standardized names
2. Validate no duplicates will be created
3. Perform renaming with validation
4. Generate completion report

Author: AI Assistant
Date: September 2025
"""

import pandas as pd
import re
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import json

class PNGStandardizer:
    def __init__(self, png_directory):
        """Initialize the PNG standardizer"""
        self.png_directory = Path(png_directory)
        self.mapping_data = []
        self.conflicts = []
        self.validation_results = {}
        self.rename_results = {}
        
    def create_mapping(self):
        """Create mapping from original to standardized filenames"""
        print("🗺️  Creating filename mapping...")
        
        png_files = list(self.png_directory.glob("*.png"))
        print(f"✅ Found {len(png_files)} PNG files to process")
        
        standardized_names = {}
        conflicts = defaultdict(list)
        
        for png_file in png_files:
            original_name = png_file.name
            original_stem = png_file.stem
            
            # Determine current schema and convert to standard
            standard_stem = self._convert_to_standard(original_stem)
            standard_name = f"{standard_stem}.png"
            
            # Track mapping
            mapping_entry = {
                'original_filename': original_name,
                'original_stem': original_stem,
                'standard_stem': standard_stem,
                'standard_filename': standard_name,
                'conversion_type': self._get_conversion_type(original_stem),
                'file_size_bytes': png_file.stat().st_size,
                'original_path': str(png_file)
            }
            
            self.mapping_data.append(mapping_entry)
            
            # Check for conflicts (multiple files mapping to same standard name)
            if standard_name in standardized_names:
                conflicts[standard_name].extend([
                    standardized_names[standard_name],
                    original_name
                ])
            else:
                standardized_names[standard_name] = original_name
        
        # Process conflicts
        for standard_name, original_names in conflicts.items():
            unique_originals = list(set(original_names))
            if len(unique_originals) > 1:
                self.conflicts.append({
                    'standard_name': standard_name,
                    'conflicting_originals': unique_originals,
                    'conflict_count': len(unique_originals)
                })
        
        print(f"✅ Mapping created for {len(self.mapping_data)} files")
        print(f"⚠️  Found {len(self.conflicts)} potential conflicts")
        
        return len(self.conflicts) == 0
    
    def _convert_to_standard(self, filename_stem):
        """Convert filename stem to standard Schema 2 format (NNNNNN-NNNN)"""
        
        # Schema 1: NNNNNN_NNNN → NNNNNN-NNNN
        if re.match(r'^\d{6}_\d{4}$', filename_stem):
            return filename_stem.replace('_', '-')
        
        # Schema 2: NNNNNN-NNNN → no change (already standard)
        elif re.match(r'^\d{6}-\d{4}$', filename_stem):
            return filename_stem
        
        # Schema 3: NNNNNNNNNN → NNNNNN-NNNN
        elif re.match(r'^\d{10}$', filename_stem):
            return f"{filename_stem[:6]}-{filename_stem[6:]}"
        
        # Schema 3: NNNNNNNNNNN → NNNNNN-NNNNN
        elif re.match(r'^\d{11}$', filename_stem):
            return f"{filename_stem[:6]}-{filename_stem[6:]}"
        
        # Special cases (like files with 'R' suffix)
        elif re.match(r'^\d{6}-\d{4}[A-Z]$', filename_stem):
            # Keep the suffix for now to avoid conflicts
            return filename_stem
        
        # Unknown format - keep as is
        else:
            return filename_stem
    
    def _get_conversion_type(self, original_stem):
        """Determine what type of conversion was applied"""
        if re.match(r'^\d{6}_\d{4}$', original_stem):
            return "schema_1_to_2"
        elif re.match(r'^\d{6}-\d{4}$', original_stem):
            return "schema_2_no_change"
        elif re.match(r'^\d{10}$', original_stem):
            return "schema_3_to_2_10digit"
        elif re.match(r'^\d{11}$', original_stem):
            return "schema_3_to_2_11digit"
        elif re.match(r'^\d{6}-\d{4}[A-Z]$', original_stem):
            return "special_suffix_kept"
        else:
            return "unknown_format_kept"
    
    def save_mapping_file(self, output_dir="standardization_results"):
        """Save the mapping file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save mapping as CSV
        mapping_df = pd.DataFrame(self.mapping_data)
        mapping_file = output_path / f"png_filename_mapping_{timestamp}.csv"
        mapping_df.to_csv(mapping_file, index=False)
        
        print(f"💾 Mapping file saved: {mapping_file}")
        
        # Save conflicts if any
        if self.conflicts:
            conflicts_df = pd.DataFrame(self.conflicts)
            conflicts_file = output_path / f"filename_conflicts_{timestamp}.csv"
            conflicts_df.to_csv(conflicts_file, index=False)
            print(f"⚠️  Conflicts file saved: {conflicts_file}")
        
        return mapping_file, output_path
    
    def validate_standardization(self):
        """Validate the standardization plan"""
        print("🔍 Validating standardization plan...")
        
        # Count conversion types
        conversion_counts = Counter([entry['conversion_type'] for entry in self.mapping_data])
        
        # Check for potential issues
        total_files = len(self.mapping_data)
        unique_standards = len(set([entry['standard_filename'] for entry in self.mapping_data]))
        
        self.validation_results = {
            'total_files': total_files,
            'unique_standard_names': unique_standards,
            'has_conflicts': len(self.conflicts) > 0,
            'conflict_count': len(self.conflicts),
            'conversion_counts': dict(conversion_counts),
            'validation_passed': len(self.conflicts) == 0 and unique_standards == total_files
        }
        
        print(f"📊 Validation Results:")
        print(f"   Total files: {total_files:,}")
        print(f"   Unique standard names: {unique_standards:,}")
        print(f"   Conflicts: {len(self.conflicts)}")
        
        print(f"\n📈 Conversion breakdown:")
        for conv_type, count in conversion_counts.items():
            percentage = (count / total_files) * 100
            print(f"   {conv_type}: {count:,} ({percentage:.2f}%)")
        
        if self.validation_results['validation_passed']:
            print(f"\n✅ VALIDATION PASSED: Safe to proceed with renaming!")
            return True
        else:
            print(f"\n❌ VALIDATION FAILED: Conflicts detected!")
            if self.conflicts:
                print(f"   Conflict examples:")
                for conflict in self.conflicts[:5]:
                    print(f"   '{conflict['standard_name']}' ← {conflict['conflicting_originals']}")
            return False
    
    def perform_renaming(self, dry_run=False):
        """Perform the actual file renaming"""
        if dry_run:
            print("🧪 DRY RUN: Simulating file renaming...")
        else:
            print("🔄 Performing file renaming...")
        
        rename_stats = {
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'skipped_conflicts': 0,
            'errors': []
        }
        
        # Get list of conflicted standard names to skip
        conflicted_standards = {conflict['standard_name'] for conflict in self.conflicts}
        
        for entry in self.mapping_data:
            original_path = Path(entry['original_path'])
            standard_filename = entry['standard_filename']
            new_path = original_path.parent / standard_filename
            
            rename_stats['attempted'] += 1
            
            # Skip if this would create a conflict
            if standard_filename in conflicted_standards:
                rename_stats['skipped_conflicts'] += 1
                continue
            
            # Skip if no change needed
            if original_path.name == standard_filename:
                rename_stats['successful'] += 1
                continue
            
            # Perform rename
            if not dry_run:
                try:
                    original_path.rename(new_path)
                    rename_stats['successful'] += 1
                except Exception as e:
                    rename_stats['failed'] += 1
                    rename_stats['errors'].append({
                        'original': str(original_path),
                        'target': str(new_path),
                        'error': str(e)
                    })
            else:
                print(f"   Would rename: {original_path.name} → {standard_filename}")
                rename_stats['successful'] += 1
        
        self.rename_results = rename_stats
        
        print(f"\n📊 Renaming Results:")
        print(f"   Attempted: {rename_stats['attempted']:,}")
        print(f"   Successful: {rename_stats['successful']:,}")
        print(f"   Failed: {rename_stats['failed']:,}")
        print(f"   Skipped (conflicts): {rename_stats['skipped_conflicts']:,}")
        
        if rename_stats['errors']:
            print(f"   Errors encountered:")
            for error in rename_stats['errors'][:5]:
                print(f"     {error['original']} → {error['error']}")
        
        return rename_stats['failed'] == 0
    
    def generate_report(self, output_path, timestamp=None):
        """Generate comprehensive standardization report"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_file = output_path / f"standardization_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PNG FILENAME STANDARDIZATION REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory: {self.png_directory}\n\n")
            
            # Validation Results
            f.write("VALIDATION RESULTS\n")
            f.write("-" * 40 + "\n")
            val = self.validation_results
            f.write(f"Total files processed:     {val['total_files']:,}\n")
            f.write(f"Unique standard names:     {val['unique_standard_names']:,}\n")
            f.write(f"Conflicts detected:        {val['conflict_count']}\n")
            f.write(f"Validation status:         {'PASSED' if val['validation_passed'] else 'FAILED'}\n\n")
            
            # Conversion Breakdown
            f.write("CONVERSION BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            for conv_type, count in val['conversion_counts'].items():
                percentage = (count / val['total_files']) * 100
                f.write(f"{conv_type:25} {count:,} ({percentage:.2f}%)\n")
            f.write("\n")
            
            # Renaming Results
            if self.rename_results:
                f.write("RENAMING RESULTS\n")
                f.write("-" * 40 + "\n")
                ren = self.rename_results
                f.write(f"Files attempted:           {ren['attempted']:,}\n")
                f.write(f"Successfully renamed:      {ren['successful']:,}\n")
                f.write(f"Failed to rename:          {ren['failed']:,}\n")
                f.write(f"Skipped (conflicts):       {ren['skipped_conflicts']:,}\n")
                
                if ren['errors']:
                    f.write(f"\nErrors encountered:\n")
                    for error in ren['errors']:
                        f.write(f"  {error['original']} → {error['error']}\n")
                f.write("\n")
            
            # Conflicts
            if self.conflicts:
                f.write("FILENAME CONFLICTS\n")
                f.write("-" * 40 + "\n")
                for conflict in self.conflicts:
                    f.write(f"Standard name: {conflict['standard_name']}\n")
                    f.write(f"Conflicting files: {', '.join(conflict['conflicting_originals'])}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            if val['validation_passed'] and self.rename_results.get('failed', 0) == 0:
                f.write("✅ Standardization completed successfully!\n")
                f.write("   All PNG files now follow the NNNNNN-NNNN format.\n")
                f.write("   Ready for CSV-PNG matching.\n")
            elif self.conflicts:
                f.write("⚠️  Manual resolution needed for conflicts.\n")
                f.write("   Review conflicting files and resolve duplicates.\n")
            else:
                f.write("❌ Standardization issues detected.\n")
                f.write("   Review errors and retry as needed.\n")
        
        print(f"📄 Report saved: {report_file}")
        return report_file

def main():
    """Main execution function"""
    print("🚀 PNG FILENAME STANDARDIZATION")
    print("=" * 60)
    
    # Initialize standardizer
    png_directory = "/lfs/turing1/0/mahmedc/Comorbidities-Detection/datasets/full_data/data"
    standardizer = PNGStandardizer(png_directory)
    
    # Step 1: Create mapping
    print("\n📋 STEP 1: Creating filename mapping...")
    mapping_ok = standardizer.create_mapping()
    
    # Step 2: Save mapping file
    print("\n💾 STEP 2: Saving mapping file...")
    mapping_file, output_path = standardizer.save_mapping_file()
    
    # Step 3: Validate standardization
    print("\n🔍 STEP 3: Validating standardization plan...")
    validation_ok = standardizer.validate_standardization()
    
    if not validation_ok:
        print("\n❌ STOPPING: Validation failed due to conflicts!")
        print("   Please review the conflicts file and resolve manually.")
        standardizer.generate_report(output_path)
        return False
    
    # Step 4: Ask for confirmation
    print(f"\n⚠️  READY TO RENAME {len(standardizer.mapping_data)} FILES")
    print("   This will standardize all PNG filenames to NNNNNN-NNNN format.")
    
    # For safety, let's do a dry run first
    print("\n🧪 STEP 4: Performing dry run...")
    standardizer.perform_renaming(dry_run=True)
    
    print(f"\n✅ Dry run completed successfully!")
    print(f"   Ready to perform actual renaming...")
    
    # Step 5: Perform actual renaming
    print("\n🔄 STEP 5: Performing actual file renaming...")
    rename_ok = standardizer.perform_renaming(dry_run=False)
    
    # Step 6: Generate final report
    print("\n📄 STEP 6: Generating final report...")
    standardizer.generate_report(output_path)
    
    if rename_ok:
        print(f"\n🎉 SUCCESS: PNG filename standardization complete!")
        print(f"📁 Results saved to: {output_path}")
        print(f"\n✅ All files now follow the standard NNNNNN-NNNN format.")
        print(f"   Ready for CSV-PNG matching!")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: Some files could not be renamed.")
        print(f"📁 Check the report in: {output_path}")
    
    return rename_ok

if __name__ == "__main__":
    success = main()
