import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.labels import Gender, Condition
from torchvision import transforms

pd.set_option('display.max_columns', None)

class ClassifierDataset(Dataset):
    """
    Load images and corresponding labels for
    """

    def __init__(self, data_path, transforms=None, size=256, train=True, age_norm=100.0, raf_norm=10, two_view=False):
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
            raise IOError(f'Data path given for ClassifierDataset does not exist: {data_path}')

        self.data_path = data_path
        self.age_norm = age_norm
        self.raf_norm = raf_norm
        self.size = size
        self.file_name = 'train.csv' if train else 'test.csv'
        self.df = pd.read_csv(os.path.join(data_path, self.file_name))

        # data cleaning
        self.df = self.df.loc[self.df['CTBiomarkers.BMDL1Values.BMDL1StandardHU'] != '--']
        self.df = self.df.loc[self.df['GENDER'].str.contains('male')] # removing cases where GENDER is NA
        # For Z-Score Normalization
        self.calcium_mean = self.df['CTBiomarkers.CalciumScoring.AbdominalAgatston'].mean()
        self.calcium_std = self.df['CTBiomarkers.CalciumScoring.AbdominalAgatston'].std()
        self.muscle_mean = self.df['CTBiomarkers.MuscleValues.L1MuscleArea'].mean()
        self.muscle_std = self.df['CTBiomarkers.MuscleValues.L1MuscleArea'].std()

        # For Min-Max Scalimg
        self.calcium_max = self.df['CTBiomarkers.CalciumScoring.AbdominalAgatston'].max()
        self.calcium_min = self.df['CTBiomarkers.CalciumScoring.AbdominalAgatston'].min()

        # Focusing on Only 1 Continuous Target for now
        self.conditions = [x for x in self.df.columns.to_list() if ('Calcium' in x)]
        self.transforms = transforms
        self.two_view = two_view

        print(f'Dataset Statistics (train = {train}):')
        print(self.df.describe(include='all'))

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
        @return tensor : tensor of condition codes at idx
        """
        data = self.df.iloc[idx]
        # tensor composition:
        # FILE,GENDER,AGE,HCC18,HCC22,...,RAF,CTBiomarkers.CalciumScoring.AbdominalAgatston,...
        t = torch.zeros(len(self.conditions), dtype=torch.float32)
        t[0] = Gender.convert(data['GENDER'])
        for i, condition in enumerate(self.conditions):
            if condition == 'GENDER':
                t[i] = Gender.convert(data['GENDER'])
            elif condition == 'AGE':
                t[i] = float(data['AGE']) / self.age_norm
            elif condition == 'RAF':
                t[i] = float(data['RAF']) / self.raf_norm
            elif condition == 'RAF':
                t[i] = float(data['RAF']) / self.raf_norm
            elif condition.startswith('HCC'):
                t[i] = Condition.convert(data[condition])
            elif condition.startswith('CTBiomarkers.CalciumScoring'):
                t[i] = (float(data[condition]) - self.calcium_min) / (self.calcium_max - self.calcium_min) # Min-Max Scaling
            elif condition.startswith('CTBiomarkers.MuscleValues'):
                t[i] = (float(data[condition]) - self.muscle_mean) / self.muscle_std
            else:
                try:
                    t[i] = float(data[condition])
                except ValueError:
                    print(data)
                    raise KeyError(condition)

        if self.two_view:
            img1 = self.get_image(data['FILE'])
            img2 = self.get_image(data['FILE'][:-4] + '-rot' + data['FILE'][-4:])
            return (img1, img2), t
        else:
            img = self.get_image(data['FILE'])
            return img, t

    def at(self, idx):
        """
        Gets directory name for a certain index
        @param idx : idx of data directory desired
        @return name : name of study at idx
        """
        return self.df.iloc[idx]['FILE'].split('.')[0]

    def get_image(self, file_name):
        img = Image.open(os.path.join(self.data_path, 'data', file_name))
        img = img.resize((self.size, self.size), Image.LANCZOS)
        img = img.convert('L')
        if self.transforms:
            img = self.transforms(img)
        return img


def get_datasets(args):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomApply([
            transforms.RandomAffine(10),
            transforms.RandomResizedCrop(256, scale=(1.0, 1.1), ratio=(0.75, 1.33)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2)
        ], p=0.75),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.75),
        transforms.ToTensor(),
        transforms.Normalize((0.55001191,), (0.18854326,))
    ])

    train_dataset = ClassifierDataset(
        args.data_dir,
        transforms=train_transform,
        size=args.size,
        train=True,
        age_norm=args.age_norm,
        raf_norm=args.raf_norm,
        two_view=args.two_view
    )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.55001191,), (0.18854326,))
    ])

    test_dataset = ClassifierDataset(
        args.data_dir,
        transforms=test_transform,
        size=args.size,
        train=False,
        age_norm=args.age_norm,
        raf_norm=args.raf_norm,
        two_view=args.two_view
    )

    return train_dataset, test_dataset
