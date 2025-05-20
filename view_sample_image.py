import os
import cv2
import matplotlib.pyplot as plt
import glob

# Dataset directory
dataset_dir = "data/kitti_sample/training/image_2/"

# Find all image paths
image_paths = sorted(glob.glob(os.path.join(dataset_dir, "*.png")))

print(f"Total images found: {len(image_paths)}")

# Select 6 evenly spaced sample images
num_samples = 6
if len(image_paths) >= num_samples:
    step = len(image_paths) // num_samples
    selected_paths = [image_paths[i * step] for i in range(num_samples)]
else:
    selected_paths = image_paths[:num_samples]

# Plot images in 2 rows x 3 columns
plt.figure(figsize=(15, 8))
for i, img_path in enumerate(selected_paths):
    img = cv2.imread(img_path)
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img_rgb)
        plt.title(f"Image {i+1}")
        plt.axis('off')
    else:
        print(f"Warning: Failed to load {img_path}")

plt.tight_layout()
plt.show()
