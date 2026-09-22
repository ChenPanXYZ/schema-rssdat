import pandas as pd
import hashlib
import os


# Function to generate a consistent hash ID
def generate_hash(val):
    return hashlib.sha256(str(val).encode('utf-8')).hexdigest()


# Load the dataset
splits = {'valid': 'data/valid-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/shangzhu/ChemQA-lite/" + splits["valid"])

questions_list = []
knowledges_list = []

# Group by the 'question' text to find duplicates/similar items
grouped = df.groupby('question')

for question_text, group in grouped:
    # 1. Process the first five as unique problems
    # We use min(len, 5) to avoid errors if a group has fewer than 5 items
    first_five = group.iloc[:10].copy()

    # Store the generated hashes to link knowledge later
    problem_hashes = []

    for _, row in first_five.iterrows():
        # ignore the sixth one
        if _ == 5:
            print("skipping sixth")
            continue
        row_dict = row.to_dict()
        # Create a unique ID based on the original row ID
        hashed_id = generate_hash(row['id'])
        row_dict['id'] = hashed_id
        problem_hashes.append(hashed_id)
        questions_list.append(row_dict)

    sixth_row = group.iloc[5]
    knowledge_base_id = generate_hash(sixth_row['id'])

    for p_hash in problem_hashes:
        k_row = sixth_row.to_dict()
        # The knowledge entry gets its own unique ID (based on original + reference)
        k_row['id'] = generate_hash(f"{sixth_row['id']}")
        # This links the knowledge to the specific question
        k_row['reference_to'] = p_hash
        k_row['reference_type'] = "similar"
        knowledges_list.append(k_row)

# Create DataFrames
df_questions = pd.DataFrame(questions_list)
df_knowledges = pd.DataFrame(knowledges_list)

# Save the files into the expected directory structure
os.makedirs('raw_files', exist_ok=True)
df_questions.to_csv("vision_questions.csv", index=False)
df_knowledges.to_csv("vision_knowledges.csv", index=False)

print(f"Created {len(df_questions)} question entries.")
print(f"Created {len(df_knowledges)} knowledge mapping entries.")
