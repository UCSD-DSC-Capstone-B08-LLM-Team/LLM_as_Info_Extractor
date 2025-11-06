# Dataset Selection for Retrieval Benchmarking

We selected MedQA as our initial benchmark dataset for information retrieval tasks. MedQA provides multiple-choice medical questions with a correct answer and explanations. 

For our retrieval benchmark:
- We treat the **question text** as the retrieval query.
- The **correct answer** is the ground-truth document.
- All other answer options are treated as distractors.
- We include all distractors dynamically to handle questions with multiple answer choices.

The dataset is now formatted as a CSV with columns:
- `question`
- `correct_answer`
- `distractor_1`, `distractor_2`, `distractor_3`, `distractor_4`

This setup provides a clean, fully ground-truthed benchmark before moving on to more realistic clinical notes in MIMIC-III, which will require additional preprocessing and ground-truth definitions.
