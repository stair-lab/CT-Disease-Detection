import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from utils.labels import *

class ClassifierDataset(Dataset):
    """
    Load images and corresponding labels for
    """
    def __init__(self, data_path, conditions, transforms=None, size=256, train=True, age_norm=101.0, raf_norm=10):
        """
        Initialize data set
        Loads and preprocesses data
        @param data_path : path to data and labels
        @param size : size of each xray
        @param train : load train or test dataset
        @param age_norm : normalization constant for age
        @param raf_norm : normalization constant for RAF
        """
        if not os.path.exists(data_path):
            raise IOError('Path given for ClassifierDataset {} does not exist...'.format(data_path))
        self.data_path = data_path

        self.age_norm = age_norm
        self.raf_norm = raf_norm
        self.size = size
        self.conditions = conditions #['HCC18', 'HCC22', 'HCC40', 'HCC48', 'HCC59', 'HCC85', 'HCC96', 'HCC108', 'HCC111', 'HCC138']
        
        # Define calcium scoring classes
        self.calcium_classes = 4  # Assuming 4 classes for calcium scoring
        
        csv_name = 'train1.csv' if train else 'test1.csv'
        self.df = pd.read_csv(os.path.join(data_path, csv_name))
        self.df['RAF'] = 0
        
        # Map calcium scoring values to class indices instead of binary
        # Assuming classes are: NONE, LOW, MEDIUM, HIGH (0-3)
        self.calcium_mapping = {
            'ABSENT': 1,
            'LOW': 1,
            'MEDIUM': 0,
            'HIGH': 0
        }
        
        self.transforms = transforms
        print(self.conditions)

    def __len__(self):
        """
        Get length of dataset
        @return len : length of dataset
        """
        return self.df.shape[0]

    def __getitem__(self,idx):
        """
        Gets data at a certain index
        @param idx : idx of data desired
        @return xray : xray image at idx
        @return tensor : tensor of condition codes at idx
        """
        data = self.df.iloc[idx]
        
        # Calculate the total tensor size:
        # 1 (gender) + len(conditions) - 1 (calcium) + calcium_classes + 2 (age, raf)
        total_tensor_size = 1 + (len(self.conditions) - 1) + self.calcium_classes + 2
        
        t = torch.zeros(total_tensor_size, dtype=torch.float32)
        
        # Set gender
        t[0] = Gender.convert(data['GENDER'])
        
        # Set binary conditions (excluding calcium scoring)
        condition_idx = 1
        for i, condition in enumerate(self.conditions, 1):
            if condition != 'CalciumScoring_AbdominalAgatston':
                t[condition_idx] = Condition.convert(data[condition])
                condition_idx += 1
        
        # One-hot encode calcium scoring
        calcium_value = data['CalciumScoring_AbdominalAgatston']
        if calcium_value in self.calcium_mapping:
            calcium_idx = self.calcium_mapping[calcium_value]
        else:
            # Default to lowest class if not found
            calcium_idx = 0
            
        # Start index for calcium one-hot encoding
        calcium_start_idx = 1 + (len(self.conditions) - 1)
        t[calcium_start_idx + calcium_idx] = 1.0
        
        # Set age and RAF
        t[-2] = float(data['AGE'])/self.age_norm
        t[-1] = float(data['RAF'])/self.raf_norm
        
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
