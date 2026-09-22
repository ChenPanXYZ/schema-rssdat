import pandas as pd
from datasets import load_dataset


def download_math_level_5():
    # 1. Properly load the dataset from Hugging Face
    print("📥 Downloading MATH-500 dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    # 2. Convert to Pandas DataFrame
    df = pd.DataFrame(dataset)

    # 3. Filter for Level 5
    # Note: Depending on the version, 'level' might be a string or integer
    # We cast to string and check if it contains '5' just to be safe
    df_level_5 = df[df['level'].astype(str) == '5']

    # 4. Save to CSV
    output_path = "math.csv"
    df_level_5.to_csv(output_path, index=False)

    print(f"✅ Saved MATH level 5 to CSV: {df_level_5.shape}")
    print(f"📍 Location: {output_path}")


if __name__ == "__main__":
    download_math_level_5()
