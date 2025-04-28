import os
import cv2
import matplotlib.pyplot as plt
import glob

# Dataset directory - corrected
dataset_dir = "data/kitti_sample/training/image_2/"

# Find all image paths
image_paths = glob.glob(os.path.join(dataset_dir, "*.png"))
image_paths = sorted(image_paths)

print(f"Total images found: {len(image_paths)}")

# Load a middle image (safe choice)
sample_img_path = image_paths[3000] if len(image_paths) > 3000 else image_paths[0]  # fallback if few images
print(f"Loading sample image: {sample_img_path}")

# Load image
sample_img = cv2.imread(sample_img_path)

# Check if image loaded correctly
if sample_img is None:
    print("Error: Failed to load image.")
else:
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(sample_img)
    plt.title("Sample KITTI Image")
    plt.axis('off')
    plt.show()
