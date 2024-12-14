import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from scipy.stats import randint
import matplotlib.pyplot as plt

# File paths
train_file = "results/train_features_combined.csv"
eval_file = "results/eval_features_combined.csv"
submission_file = "results/final/eval_results2.csv"  # Output predictions for evaluation data
train_predictions_file = "results/final/train_results2.csv"  # Output predictions for training data

# Load training data
print("Loading training data...")
train_df = pd.read_csv(train_file)

# Separate features and target
X_train = train_df.drop(columns=["EventType"])  # Use all columns except MatchID and target
y_train = train_df["EventType"]

# Handle class imbalance with SMOTE
print("Balancing data with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Define the parameter grid
param_grid = {
    'n_estimators': randint(100, 500),               # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],                 # Maximum depth of the trees
    'min_samples_split': randint(2, 10),            # Minimum number of samples required to split an internal node
    'min_samples_leaf': randint(1, 10),             # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt', 'log2', None],         # Number of features to consider when looking for the best split
    'bootstrap': [True, False]                      # Whether bootstrap samples are used when building trees
}
# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Perform Randomized Search
print("Starting Randomized Search for Hyperparameter Tuning...")
stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,
    scoring='accuracy',
    cv=stratified_cv,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_balanced, y_train_balanced)

# Best parameters from RandomizedSearchCV
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Train final Random Forest model
print("Training final Random Forest model...")
best_rf = RandomForestClassifier(**best_params, random_state=42)
best_rf.fit(X_train_balanced, y_train_balanced)

# Feature importance
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
feature_importances.sort_values(ascending=False).plot(kind='barh', figsize=(10, 6))
plt.title("Top Feature Importances")
plt.show()

# Load evaluation data
print("Loading evaluation data...")
eval_df = pd.read_csv(eval_file)

# Predict for evaluation data
eval_df["EventType"] = best_rf.predict(eval_df)  


# Generate submission file
print("Generating submission file...")
submission_df = eval_df[["ID", "EventType"]]  
submission_df.to_csv(submission_file, index=False)

print(f"Submission file saved: {submission_file}")

# Generate predictions for training data
print("Generating predictions for training data...")
train_df["EventType"] = best_rf.predict(X_train)

# Save the predictions for training data
train_results = train_df[["ID", "EventType"]]
train_results.to_csv(train_predictions_file, index=False)
print(f"Training predictions saved: {train_predictions_file}")

# Evaluate performance on the training set
print("Evaluating performance on the training set...")
y_train_pred = best_rf.predict(X_train_balanced)
print("Training Accuracy:", accuracy_score(y_train_balanced, y_train_pred))
print("Classification Report on Training Data:")
print(classification_report(y_train_balanced, y_train_pred))
