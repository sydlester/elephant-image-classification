from torch.utils.data import Dataset
from PIL import Image
import requests
from io import BytesIO
import torch

# Create a custom pytorch dataset for the images and labels
class CustomImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_url = self.df.iloc[idx, 0] 
        try:
            response = requests.get(img_url)         
        except:
            # remove entry from dataloader
            print(f"Removing image {img_url} due to error")
            self.df.drop(self.df.index[idx], inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            return self.__getitem__(idx)
        
        if response.status_code != 200:
            # remove entry from dataloader
            print(f"Removing image {img_url} due to error code {response.status_code}")
            self.df.drop(self.df.index[idx], inplace=True)
            self.df.reset_index(drop=True, inplace=True)

            return self.__getitem__(idx)
        
        image = Image.open(BytesIO(response.content))
        label = self.df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label