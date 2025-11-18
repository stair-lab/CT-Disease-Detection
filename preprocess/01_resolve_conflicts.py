#!/usr/bin/env python3
"""
Resolve PNG filename conflicts by analyzing duplicate files
"""

import pandas as pd
from pathlib import Path
import hashlib

def analyze_conflicts():
    print("🔍 Analyzing filename conflicts...")
    
    # Read conflicts file
    conflicts_file = "standardization_results/filename_conflicts_20250904_045330.csv"
    conflicts_df = pd.read_csv(conflicts_file)
    
    png_directory = Path("/lfs/skampere2/0/mahmedc/Comorbidities-Detection/datasets/full_data/data")
    
    print(f"📊 Found {len(conflicts_df)} conflicts to resolve")
    
    resolution_plan = []
    
    for _, conflict in conflicts_df.iterrows():
        standard_name = conflict['standard_name']
        conflicting_files = eval(conflict['conflicting_originals'])  # Convert string back to list
        
        print(f"\n🔍 Analyzing conflict: {standard_name}")
        print(f"   Conflicting files: {conflicting_files}")
        
        file_info = []
        for filename in conflicting_files:
            file_path = png_directory / filename
            if file_path.exists():
                file_size = file_path.stat().st_size
                
                # Calculate file hash to check if files are identical
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                
                file_info.append({
                    'filename': filename,
                    'size_bytes': file_size,
                    'hash': file_hash,
                    'exists': True
                })
            else:
                file_info.append({
                    'filename': filename,
                    'size_bytes': 0,
                    'hash': 'N/A',
                    'exists': False
                })
        
        # Analyze the files
        existing_files = [f for f in file_info if f['exists']]
        
        if len(existing_files) == 0:
            print("   ❌ No files exist!")
            continue
        elif len(existing_files) == 1:
            print(f"   ✅ Only one file exists: {existing_files[0]['filename']}")
            resolution_plan.append({
                'conflict_group': standard_name,
                'resolution': 'keep_only_existing',
                'keep_file': existing_files[0]['filename'],
                'remove_files': []
            })
        else:
            # Check if files are identical
            hashes = [f['hash'] for f in existing_files]
            sizes = [f['size_bytes'] for f in existing_files]
            
            if len(set(hashes)) == 1:
                print("   ✅ Files are identical (same hash)")
                # Keep the hyphen version (Schema 2 standard), remove underscore version
                hyphen_file = None
                underscore_files = []
                
                for f in existing_files:
                    if '-' in f['filename']:
                        hyphen_file = f['filename']
                    else:
                        underscore_files.append(f['filename'])
                
                if hyphen_file:
                    print(f"   📌 Keeping hyphen version: {hyphen_file}")
                    print(f"   🗑️  Removing underscore versions: {underscore_files}")
                    resolution_plan.append({
                        'conflict_group': standard_name,
                        'resolution': 'keep_hyphen_remove_underscore',
                        'keep_file': hyphen_file,
                        'remove_files': underscore_files
                    })
                else:
                    print("   ⚠️  No hyphen version found - manual review needed")
                    resolution_plan.append({
                        'conflict_group': standard_name,
                        'resolution': 'manual_review_needed',
                        'keep_file': existing_files[0]['filename'],  # Keep first one for now
                        'remove_files': [f['filename'] for f in existing_files[1:]]
                    })
            else:
                print("   ⚠️  Files are different (different hashes)")
                print("   📏 File sizes:")
                for f in existing_files:
                    print(f"      {f['filename']}: {f['size_bytes']:,} bytes")
                
                # Keep the larger file (assuming it's more complete)
                largest_file = max(existing_files, key=lambda x: x['size_bytes'])
                other_files = [f['filename'] for f in existing_files if f['filename'] != largest_file['filename']]
                
                print(f"   📌 Keeping largest file: {largest_file['filename']} ({largest_file['size_bytes']:,} bytes)")
                print(f"   🗑️  Removing smaller files: {other_files}")
                
                resolution_plan.append({
                    'conflict_group': standard_name,
                    'resolution': 'keep_largest_file',
                    'keep_file': largest_file['filename'],
                    'remove_files': other_files
                })
    
    return resolution_plan

def execute_resolution(resolution_plan, dry_run=True):
    """Execute the conflict resolution plan"""
    if dry_run:
        print("\n🧪 DRY RUN: Simulating conflict resolution...")
    else:
        print("\n🔄 Executing conflict resolution...")
    
    png_directory = Path("/lfs/skampere2/0/mahmedc/Comorbidities-Detection/datasets/full_data/data")
    
    stats = {
        'files_removed': 0,
        'files_kept': 0,
        'errors': []
    }
    
    for plan in resolution_plan:
        print(f"\n📋 Resolving: {plan['conflict_group']}")
        print(f"   Strategy: {plan['resolution']}")
        print(f"   Keep: {plan['keep_file']}")
        print(f"   Remove: {plan['remove_files']}")
        
        # Keep file (no action needed)
        stats['files_kept'] += 1
        
        # Remove conflicting files
        for remove_file in plan['remove_files']:
            file_path = png_directory / remove_file
            
            if not dry_run:
                try:
                    if file_path.exists():
                        file_path.unlink()  # Delete the file
                        print(f"   ✅ Removed: {remove_file}")
                        stats['files_removed'] += 1
                    else:
                        print(f"   ⚠️  File not found: {remove_file}")
                except Exception as e:
                    print(f"   ❌ Error removing {remove_file}: {e}")
                    stats['errors'].append({'file': remove_file, 'error': str(e)})
            else:
                print(f"   🗑️  Would remove: {remove_file}")
                stats['files_removed'] += 1
    
    print(f"\n📊 Resolution Results:")
    print(f"   Files kept: {stats['files_kept']}")
    print(f"   Files removed: {stats['files_removed']}")
    print(f"   Errors: {len(stats['errors'])}")
    
    return len(stats['errors']) == 0

def main():
    print("🚀 PNG FILENAME CONFLICT RESOLUTION")
    print("=" * 60)
    
    # Analyze conflicts
    resolution_plan = analyze_conflicts()
    
    if not resolution_plan:
        print("\n❌ No conflicts found or unable to create resolution plan")
        return False
    
    # Show resolution plan
    print(f"\n📋 RESOLUTION PLAN:")
    print("=" * 60)
    for i, plan in enumerate(resolution_plan, 1):
        print(f"{i}. {plan['conflict_group']}")
        print(f"   Strategy: {plan['resolution']}")
        print(f"   Keep: {plan['keep_file']}")
        if plan['remove_files']:
            print(f"   Remove: {', '.join(plan['remove_files'])}")
        print()
    
    # Execute dry run
    print("🧪 Performing dry run...")
    dry_run_ok = execute_resolution(resolution_plan, dry_run=True)
    
    if not dry_run_ok:
        print("❌ Dry run failed!")
        return False
    
    print("\n✅ Dry run successful!")
    
    # Execute actual resolution
    print("\n🔄 Executing actual conflict resolution...")
    actual_ok = execute_resolution(resolution_plan, dry_run=False)
    
    if actual_ok:
        print("\n🎉 SUCCESS: All conflicts resolved!")
        print("   Ready to re-run standardization script.")
        return True
    else:
        print("\n❌ Some errors occurred during resolution.")
        return False

if __name__ == "__main__":
    success = main()
