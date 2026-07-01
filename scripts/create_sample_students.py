from pathlib import Path
import pandas as pd
import random

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]

QUESTIONS_FILE = ROOT / "sample_data" / "questions.csv"

STUDENTS_FILE = ROOT / "sample_data" / "students.csv"
RESPONSES_FILE = ROOT / "sample_data" / "responses.csv"

# ----------------------------------------------------------
# Load Questions
# ----------------------------------------------------------

questions = pd.read_csv(QUESTIONS_FILE)

# ----------------------------------------------------------
# Create Sample Students
# ----------------------------------------------------------

students = pd.DataFrame([
    {
        "student_id": "S1",
        "previous_percentage": 92,
        "iq_score": 118,
    },
    {
        "student_id": "S2",
        "previous_percentage": 87,
        "iq_score": 110,
    },
    {
        "student_id": "S3",
        "previous_percentage": 76,
        "iq_score": 96,
    },
    {
        "student_id": "S4",
        "previous_percentage": 63,
        "iq_score": 82,
    },
    {
        "student_id": "S5",
        "previous_percentage": 48,
        "iq_score": 72,
    }
])

students.to_csv(STUDENTS_FILE, index=False)

print(f"Created {STUDENTS_FILE}")

# ----------------------------------------------------------
# Generate Responses
# ----------------------------------------------------------

responses = []

# Probability of answering correctly
student_strength = {
    "S1": 0.92,
    "S2": 0.80,
    "S3": 0.60,
    "S4": 0.40,
    "S5": 0.20,
}

# Bloom difficulty penalty
difficulty_penalty = {
    "remember": 0.00,
    "understand": 0.05,
    "apply": 0.15,
    "analyze": 0.25,
    "evaluate": 0.35,
    "create": 0.45,
}

for _, q in questions.iterrows():

    bloom = q["bloom_level"]

    penalty = difficulty_penalty.get(bloom, 0.20)

    for student in students["student_id"]:

        base = student_strength[student]

        probability = max(
            0.05,
            min(
                0.95,
                base - penalty
            )
        )

        correct = 1 if random.random() < probability else 0

        responses.append({
            "student_id": student,
            "question_id": q["question_id"],
            "is_correct": correct
        })

responses = pd.DataFrame(responses)

responses.to_csv(
    RESPONSES_FILE,
    index=False
)

print(f"Created {RESPONSES_FILE}")

print()

print("Students Preview")
print(students)

print()

print("Responses Preview")
print(responses.head(20))