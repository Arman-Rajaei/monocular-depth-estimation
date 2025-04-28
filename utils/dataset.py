import os
import glob
from torch.utils.data import Dataset
import cv2
import torch

class KITTITinyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load the image
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 🚨 New: Resize to fixed size
        image = cv2.resize(image, (416, 128))  # (width=416, height=128)

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # Normalize between 0 and 1
        
        if self.transform:
            image = self.transform(image)
        
        return image

