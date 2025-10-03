import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn.cluster import KMeans

image_path = 'D:\Projects\GraphST/1.png'
import cv2
# Load both images
original_image = cv2.imread(image_path)  # First uploaded image
target_palette_image = cv2.imread("D:\Projects\GraphST\Images/ACT.png")  # Second uploaded image

# Convert images to RGB format
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
target_palette_image = cv2.cvtColor(target_palette_image, cv2.COLOR_BGR2RGB)

# Resize target palette image for faster processing
target_palette_resized = cv2.resize(target_palette_image, (100, 100))  # Small size to extract colors

# Extract dominant colors from the target palette using KMeans
pixels_target = target_palette_resized.reshape(-1, 3)

# Use KMeans to find 6 dominant colors (based on legend)
kmeans_target = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans_target.fit(pixels_target)

# Get the dominant colors from the target image
target_colors = kmeans_target.cluster_centers_.astype(int)

# Extract dominant colors from the original image
pixels_original = original_image.reshape(-1, 3)
kmeans_original = KMeans(n_clusters=7, random_state=42, n_init=10)
kmeans_original.fit(pixels_original)

# Get the dominant colors from the original image
original_colors = kmeans_original.cluster_centers_.astype(int)

# Create a mapping between original colors and target colors
color_mapping = {tuple(original_colors[i]): tuple(target_colors[i % 6]) for i in range(len(original_colors))}

# Replace colors in the original image
new_image = original_image.copy()
for i in range(new_image.shape[0]):
    for j in range(new_image.shape[1]):
        pixel_tuple = tuple(new_image[i, j])
        if pixel_tuple in color_mapping:
            new_image[i, j] = color_mapping[pixel_tuple]

# Display the recolored image
plt.figure(figsize=(10, 5))
plt.imshow(new_image)
plt.axis("off")
plt.title("Recolored Image to Match Target Palette")
plt.show()
