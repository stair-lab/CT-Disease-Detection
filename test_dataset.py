import os
from torch.utils.data import Dataset
from PIL import Image

class TestDataset(Dataset):
    """
    Prediction dataset that loads only input images.
    """
    def __init__(self, data_path, transforms=None, size=256):
        """
        Initialize dataset.
        @param data_path : path to image directory
        @param size : resized image dimension (square)
        """
        if not os.path.exists(data_path):
            raise IOError('Path given for TestDataset {} does not exist...'.format(data_path))
        self.data_path = data_path
        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        self.data = sorted(
            fname for fname in os.listdir(data_path)
            if fname.lower().endswith(valid_exts)
        )
        self.size = size

        self.transforms = transforms

    def __len__(self):
        """
        Get length of dataset
        @return len : length of dataset
        """
        return len(self.data)

    def __getitem__(self,idx):
        """
        Gets image at a certain index.
        @param idx : idx of data desired
        @return xray : xray image at idx
        """
        fname = self.data[idx]
        xray = Image.open(os.path.join(self.data_path, fname))
        xray = xray.resize((self.size, self.size), Image.LANCZOS)
        xray = xray.convert('L')
        if self.transforms:
            xray = self.transforms(xray)

        return xray

    def at(self,idx):
        """
        Gets directory name for a certain index
        @param idx : idx of data directory desired
        @return name : name of study at idx
        """
        return self.data[idx]
