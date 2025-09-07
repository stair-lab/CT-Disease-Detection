import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.labels import *
from config.biomarker_config import BiomarkerConfig

class ClassifierDataset(Dataset):
    """
    Load images and corresponding labels for
    """
    def __init__(self, data_path, biomarker_config, transforms=None, size=256, train=True):
        """
        Initialize data set
        Loads and preprocesses data
        @param data_path : path to data and labels
        @param biomarker_config : BiomarkerConfig object specifying which biomarkers to use
        @param size : size of each xray
        @param train : load train or test dataset
        """
        if not os.path.exists(data_path):
            raise IOError('Path given for ClassifierDataset {} does not exist...'.format(data_path))
        self.data_path = data_path
        self.size = size
        self.biomarker_config = biomarker_config
        
        csv_name = 'train1.csv' if train else 'test1.csv'
        self.df = pd.read_csv(os.path.join(data_path, csv_name))
        
        # Handle RAF column if it doesn't exist
        if 'RAF' not in self.df.columns:
            self.df['RAF'] = 0
        
        self.transforms = transforms
        
        # Get tensor layout for efficient indexing
        self.tensor_layout = self.biomarker_config.get_tensor_layout()
        
        print(f"Biomarkers configured: {self.biomarker_config.all_biomarker_names}")
        print(f"Total output tensor size: {self.biomarker_config.total_output_size}")

    def __len__(self):
        """
        Get length of dataset
        @return len : length of dataset
        """
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Gets data at a certain index
        @param idx : idx of data desired
        @return xray : xray image at idx
        @return tensor : tensor of biomarker values at idx
        """
        data = self.df.iloc[idx]
        
        # Create tensor with the configured size
        t = torch.zeros(self.biomarker_config.total_output_size, dtype=torch.float32)
        
        # Process binary biomarkers
        for biomarker in self.biomarker_config.binary_biomarkers:
            if biomarker.name in data:
                layout = self.tensor_layout[biomarker.name]
                idx_start = layout['start_idx']
                
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
                idx_start = layout['start_idx']
                
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
                idx_start = layout['start_idx']
                
                # Normalize the continuous value
                raw_value = float(data[biomarker.name])
                normalized_value = biomarker.normalize(raw_value)
                t[idx_start] = normalized_value
        
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
