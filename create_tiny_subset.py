import os
import random
import shutil
import glob

# Paths
original_data_dir = "data/kitti_sample/training/image_2/"
tiny_data_dir = "data/kitti_tiny/training/image_2/"

# Create tiny dataset folders
os.makedirs(tiny_data_dir, exist_ok=True)

# Find all image files
all_images = glob.glob(os.path.join(original_data_dir, "*.png"))

# Shuffle and pick 100 random images
random.seed(42)  # for reproducibility
tiny_images = random.sample(all_images, 100)

# Copy images to tiny dataset
for img_path in tiny_images:
    img_name = os.path.basename(img_path)
    shutil.copy(img_path, os.path.join(tiny_data_dir, img_name))

print(f"Copied {len(tiny_images)} images to {tiny_data_dir}")
