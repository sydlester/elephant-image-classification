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
        print(f"Number of images: {len(self.image_paths)}")
        print(f"Sample image path: {self.image_paths[0]}")
        image = Image.open(self.image_paths[0])
        print(self.transform(image))
        print(f"Image size: {image.size}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
        label = self.df.iloc[idx, 1]

        return image, label
