import os
import json
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(BASE_DIR, "medqa_data", "questions", "US", "train.jsonl")
output_file = os.path.join(BASE_DIR, "medqa_data", "medqa_train.csv")

rows = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        question = item["question"]
        options = item["options"]
        answer_idx = item["answer_idx"]

        correct_answer = options[answer_idx]
        distractors = [opt for i, opt in enumerate(options) if opt != answer_idx]

        # dynamically handle distractors
        row = {
            "question": question,
            "correct_answer": correct_answer
        }
        for i, d in enumerate(distractors):
            row[f"distractor_{i+1}"] = options[d]
        rows.append(row)

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Created {output_file} with {len(rows)} QA items")
