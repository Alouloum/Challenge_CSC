import pandas as pd

# File paths for the submission files
submission_file_1 = "results/final/eval_results1.csv"
submission_file_2 = "results/final/eval_results2.csv"
updated_submission_file = "results/final/submission_set_to_1.csv"

# Load the submission files
print("Loading submission files...")
sub1 = pd.read_csv(submission_file_1)
sub2 = pd.read_csv(submission_file_2)

# Check if both files have the same columns
if not sub1.columns.equals(sub2.columns):
    raise ValueError("The columns of the two submission files do not match.")

# Sort both files by the 'ID' column
print("Sorting submission files by ID...")
sub1 = sub1.sort_values(by="ID").reset_index(drop=True)
sub2 = sub2.sort_values(by="ID").reset_index(drop=True)

# Check if both files have the same IDs after sorting
if not sub1['ID'].equals(sub2['ID']):
    raise ValueError("The IDs in the two submission files do not match after sorting.")

# Find mismatched rows
print("Finding mismatched rows...")
differences = sub1[sub1["EventType"] != sub2["EventType"]]

# Set random 50% of mismatched rows' EventType in the first submission file to 1
print("Updating mismatched rows to 1...")

# Randomly select 50% of the mismatched rows
mismatched_rows = differences.sample(frac=0.5, random_state=44)

# Update the EventType of the selected rows to 1
sub1.loc[mismatched_rows.index, "EventType"] = 1

# Save the updated submission file
sub1.to_csv(updated_submission_file, index=False)
print(f"Updated submission file saved: {updated_submission_file}")
