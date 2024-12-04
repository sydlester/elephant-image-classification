import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df
        self.root_dir = root_dir
        self.image_paths = [
            os.path.join(root_dir, url) for url in df['URL']
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = Image.open(img_path)
        except Exception as e:
            print(f'Error loading image: {img_path}')
            print(e)
            return None
        
        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx, 1]

        return image, label
