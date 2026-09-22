import pandas as pd
import hashlib
import json

def generate_hash(val):
    """
    Deterministic hash for strings and simple types.
    Handles None and non-string types safely.
    """
    if val is None:
        return None
    return hashlib.sha256(str(val).encode('utf-8')).hexdigest()

# 1. Configuration
dataset_base = "hf://datasets/yujunzhou/LabSafety_Bench/"
splits = {
    'QA': 'data/QA-00000-of-00001.parquet',
    'QA_I': 'data/QA_I-00000-of-00001.parquet'
}

# 2. Load the QA_I split (this specifically contains the image-based questions)
print("Loading dataset...")
df = pd.read_parquet(dataset_base + splits["QA_I"])

# 3. Filter for rows where 'Image Path' is not null/empty
# In this dataset, images are often stored in 'Image Path' or 'image' columns
df_with_images = df[df['Image Path'].notna()].copy()

# 4. Generate a unique ID/Hash for each image path for tracking
df_with_images['image_hash'] = df_with_images['Image Path'].apply(generate_hash)

# 5. Display results
print(f"Total rows in QA_I split: {len(df)}")
print(f"Rows containing valid Image Paths: {len(df_with_images)}")

# Show the first few results including the path and our new hash
print("\n--- Samples with Image Paths ---")
print(df_with_images[['Question', 'Image Path', 'image_hash']].head())

# Optional: Save these specific rows to a new file
# df_with_images.to_csv("lab_safety_images.csv", index=False)
