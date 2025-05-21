import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class KITTITinyDataset(Dataset):
    def __init__(self, image_root, depth_root, transform=None):
        # Load all paths
        all_image_paths = sorted(glob.glob(os.path.join(image_root, "*.png")))
        all_depth_paths = sorted(glob.glob(os.path.join(depth_root, "*.png")))

        # Only use as many RGB images as depth maps, starting from the 5th
        offset = 5
        num_pairs = min(len(all_depth_paths), len(all_image_paths) - offset)

        self.image_paths = all_image_paths[offset:offset + num_pairs]
        self.depth_paths = all_depth_paths[:num_pairs]

        print(f"📸 Total RGB images before slicing: {len(all_image_paths)}")
        print(f"💡 Using {len(self.image_paths)} RGB images starting from index {offset}")
        print(f"🌊 Using {len(self.depth_paths)} depth maps")
        
        assert len(self.image_paths) == len(self.depth_paths), \
            "Mismatch between number of RGB images and depth maps"

        self.transform = transform or T.Compose([
            T.Resize((128, 416)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        depth = Image.open(self.depth_paths[idx])

        image = self.transform(image)
        depth = self.transform(depth) / 256.0  # Convert from 0–65535 to meters

        return image, depth
