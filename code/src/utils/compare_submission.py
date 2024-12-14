import pandas as pd

# File paths for the submission files
submission_file_1 = "results/final/submission_0.76171.csv"
submission_file_2 = "results/final/submission_set_to_1.csv"

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

# Compare EventType predictions
print("Comparing EventType predictions...")
differences = sub1[sub1["EventType"] != sub2["EventType"]]

# Calculate basic statistics
total_rows = len(sub1)
num_differences = len(differences)
difference_percentage = (num_differences / total_rows) * 100

print(f"Total Rows: {total_rows}")
print(f"Number of Differences: {num_differences}")
print(f"Percentage of Differences: {difference_percentage:.2f}%")

# Calculate the percentage of mismatched rows labeled `0` in the first file
labeled_0_in_first = differences[differences["EventType"] == 0]
percentage_0_in_first = (len(labeled_0_in_first) / num_differences) * 100

# Calculate the percentage of mismatched rows labeled `1` in the first file
labeled_1_in_first = differences[differences["EventType"] == 1]
percentage_1_in_first = (len(labeled_1_in_first) / num_differences) * 100

print(f"Percentage of mismatched rows labeled 0 in the first file: {percentage_0_in_first:.2f}%")
print(f"Percentage of mismatched rows labeled 1 in the first file: {percentage_1_in_first:.2f}%")
