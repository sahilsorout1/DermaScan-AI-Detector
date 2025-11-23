import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

# 1. Setup Paths
base_dir = 'dataset'
image_dir = os.path.join(base_dir, 'images')
mask_dir = os.path.join(base_dir, 'masks')
csv_path = os.path.join(base_dir, 'HAM10000_metadata.csv')

# 2. Load the Excel File (CSV)
try:
    df = pd.read_csv(csv_path)
    print(f"✅ CSV Loaded! Found {len(df)} rows of data.")
except FileNotFoundError:
    print("❌ Error: Could not find HAM10000_metadata.csv in 'dataset' folder.")
    exit()

# 3. Check one random image
row = df.sample(1).iloc[0]  # Pick a random patient
image_id = row['image_id']

# Construct file paths
img_path = os.path.join(image_dir, image_id + '.jpg')

# Note: Mask filenames in this dataset usually have '_segmentation' added
mask_path = os.path.join(mask_dir, image_id + '_segmentation.png') 

print(f"\nChecking Patient ID: {image_id}")
print(f"Looking for Image at: {img_path}")
print(f"Looking for Mask at: {mask_path}")

# 4. Load and Show
if os.path.exists(img_path) and os.path.exists(mask_path):
    print("\n✅ SUCCESS! Image and Mask both found.")
    
    # Read images
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert to standard colors
    
    mask = cv2.imread(mask_path)
    
    # Plot them side by side
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.title(f"Original: {row['dx']}") # Show cancer type
    plt.imshow(img)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Segmentation Mask")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    
    plt.show()
    print("Check the popup window to see the images!")

else:
    print("\n❌ ERROR: Files missing.")
    if not os.path.exists(img_path):
        print(f"   -> Missing Image: {img_path}")
    if not os.path.exists(mask_path):
        print(f"   -> Missing Mask: {mask_path}")
    print("Please check your folder names in 'dataset' exactly match 'images' and 'masks'.")
