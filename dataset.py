import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.labels import Condition
from config.biomarker_config import FlexibleBiomarkerConfig

class ClassifierDataset(Dataset):
    """
    Load images and corresponding labels for
    """
    def __init__(self, data_path, biomarker_config, transforms=None, size=256, train=True):
        """
        Initialize data set
        Loads and preprocesses data
        @param data_path : path to data and labels
        @param biomarker_config : FlexibleBiomarkerConfig object specifying which biomarkers to use
        @param size : size of each xray
        @param train : load train or test dataset
        """
        if not os.path.exists(data_path):
            raise IOError('Path given for ClassifierDataset {} does not exist...'.format(data_path))
        self.data_path = data_path
        self.size = size
        self.biomarker_config = biomarker_config
        
        csv_name = 'train.csv' if train else 'val.csv'
        self.df = pd.read_csv(os.path.join(data_path, csv_name))
        
        # Apply age filtering for HIPAA compliance
        self.df = self._filter_age_records()
        
        self.transforms = transforms
        
        # Get tensor layout for efficient indexing
        self.tensor_layout = self.biomarker_config.get_tensor_layout()
        
        # Pre-compute all target tensors for efficient access (needed for class weights)
        self.targets = self._prepare_all_targets()
        
        print(f"Biomarkers configured: {self.biomarker_config.get_all_biomarker_names()}")
        print(f"Total output tensor size: {self.biomarker_config.total_output_size}")

    def _filter_age_records(self):
        """
        Filter out records with AGE = "90+" for HIPAA compliance.
        Also ensures that remaining records have max age of 89.
        """
        if 'AGE' not in self.df.columns:
            print("AGE column not found - skipping age filtering")
            return self.df
        
        original_count = len(self.df)
        
        # Filter out "90+" records
        age_90_plus_mask = self.df['AGE'] == '90+'
        age_90_plus_count = age_90_plus_mask.sum()
        
        if age_90_plus_count > 0:
            print(f"HIPAA Compliance: Filtering out {age_90_plus_count:,} records with AGE='90+'")
            self.df = self.df[~age_90_plus_mask].copy()
        
        # Convert remaining AGE values to numeric and verify max age is 89
        numeric_age_mask = pd.to_numeric(self.df['AGE'], errors='coerce').notna()
        if not numeric_age_mask.all():
            # Handle any non-numeric age values (shouldn't happen after filtering 90+)
            non_numeric_count = (~numeric_age_mask).sum()
            print(f"Found {non_numeric_count} non-numeric AGE values, filtering them out")
            self.df = self.df[numeric_age_mask].copy()
        
        # Convert to numeric and verify max age
        self.df['AGE'] = pd.to_numeric(self.df['AGE'], errors='coerce')
        
        if len(self.df) > 0:
            max_age = self.df['AGE'].max()
            min_age = self.df['AGE'].min()
            
            if max_age > 89:
                print(f"Warning: Maximum age is {max_age}, expected <= 89")
            else:
                print(f"Age range after filtering: {min_age:.0f} - {max_age:.0f} years")
        
        filtered_count = len(self.df)
        removed_count = original_count - filtered_count
        
        if removed_count > 0:
            print(f"Dataset filtering summary:")
            print(f" Original records: {original_count:,}")
            print(f" Removed records: {removed_count:,}")
            print(f" Remaining records: {filtered_count:,}")
            print(f" Removal rate: {removed_count/original_count*100:.1f}%")
        
        return self.df

    def __len__(self):
        """
        Get length of dataset
        @return len : length of dataset
        """
        return self.df.shape[0]

    def _prepare_all_targets(self):
        """Pre-compute all target tensors for the dataset"""
        import numpy as np
        
        targets = []
        for idx in range(len(self.df)):
            data = self.df.iloc[idx]
            
            # Create tensor with the configured size
            t = torch.zeros(self.biomarker_config.total_output_size, dtype=torch.float32)
            
            # Process binary biomarkers
            for biomarker in self.biomarker_config.binary_biomarkers:
                if biomarker.name in data:
                    layout = self.tensor_layout[biomarker.name]
                    idx_start = layout.start_idx
                    
                    # Convert using configured classes or default Condition enum
                    if biomarker.positive_class == "PRESENT" and biomarker.negative_class == "ABSENT":
                        # Use default Condition converter
                        t[idx_start] = Condition.convert(data[biomarker.name])
                    else:
                        # Use custom class mapping
                        if data[biomarker.name] == biomarker.positive_class:
                            t[idx_start] = 1.0
                        elif data[biomarker.name] == biomarker.negative_class:
                            t[idx_start] = 0.0
                        else:
                            # Default to negative class for unknown values
                            t[idx_start] = 0.0
            
            # Process multiclass biomarkers
            for biomarker in self.biomarker_config.multiclass_biomarkers:
                if biomarker.name in data:
                    layout = self.tensor_layout[biomarker.name]
                    idx_start = layout.start_idx
                    
                    # Get class index and create one-hot encoding
                    try:
                        class_idx = biomarker.class_to_index(data[biomarker.name])
                        t[idx_start + class_idx] = 1.0
                    except ValueError:
                        # Default to first class if unknown value
                        print(f"Warning: Unknown value '{data[biomarker.name]}' for {biomarker.name}, using first class")
                        t[idx_start] = 1.0
            
            # Process continuous biomarkers
            for biomarker in self.biomarker_config.continuous_biomarkers:
                if biomarker.name in data:
                    layout = self.tensor_layout[biomarker.name]
                    idx_start = layout.start_idx
                    
                    # Normalize the continuous value
                    raw_value = float(data[biomarker.name])
                    normalized_value = biomarker.normalize(raw_value)
                    t[idx_start] = normalized_value
            
            targets.append(t.numpy())
        
        return np.array(targets)

    def __getitem__(self, idx):
        """
        Gets data at a certain index
        @param idx : idx of data desired
        @return xray : xray image at idx
        @return tensor : tensor of biomarker values at idx
        """
        data = self.df.iloc[idx]
        
        # Get pre-computed targets
        t = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        # Load and process image
        xray = Image.open(os.path.join(self.data_path, 'data', data['FILE'] + '.png'))
        xray = xray.resize((self.size, self.size), Image.LANCZOS)
        xray = xray.convert('L')
        if self.transforms:
            xray = self.transforms(xray)
        
        return xray, t

    def at(self,idx):
        """
        Gets directory name for a certain index
        @param idx : idx of data directory desired
        @return name : name of study at idx
        """
        return self.df.iloc[idx]['FILE'].split('.')[0]

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    c = ClassifierDataset('data')
    print(len(c))
    print(c[0][0].shape)
    print(c[0][1].shape)
    print(c[0][1])
    print(c.at(2))
    data = DataLoader(c, batch_size=4, shuffle=True)
    for s in data:
        print(s[0][0].shape, s[1][0], s[1][0].shape)
        print(s[0][1].shape, s[1][1], s[1][1].shape)
        print(s[0][2].shape, s[1][2], s[1][2].shape)
        print(s[0][3].shape, s[1][3], s[1][3].shape)
        break
