import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "data" / "Synapse_Quiz_30Q.xlsx"
OUTPUT_FILE = ROOT / "sample_data" / "questions.csv"

# Row 2 contains headers
df = pd.read_excel(INPUT_FILE, header=1)

# Rename columns
df = df.rename(columns={
    "Q#": "question_id",
    "Chapter": "chapter",
    "concept_id": "concept_id",
    "Bloom": "bloom_level",
    "Difficulty": "difficulty",
    "Type": "question_type",
    "Correct Answer": "correct_answer",
    "Correct Reasoning": "correct_reasoning",
})

required = [
    "question_id",
    "chapter",
    "concept_id",
    "bloom_level",
    "difficulty",
    "question_type",
    "correct_answer",
    "correct_reasoning",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

questions = df[required].copy()

# Standardize Bloom labels
questions["bloom_level"] = (
    questions["bloom_level"]
    .astype(str)
    .str.strip()
    .str.lower()
)

questions.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"\nSaved {len(questions)} questions")
print(OUTPUT_FILE)
print("\nPreview:\n")
print(questions.head())