import pandas as pd
import sys
from sklearn.metrics import accuracy_score, f1_score
import os

# =========================
# CONFIGURATION
# =========================
metric = "Accuracy"   # or "F1-Score"

# =========================
# Arguments
# =========================
if len(sys.argv) < 3:
    print("Usage: python evaluate.py <submission_csv> <labels_csv>")
    sys.exit(1)

submission_csv = sys.argv[1]
labels_csv = sys.argv[2]

# =========================
# GitHub username
# =========================
github_user = os.environ.get("GITHUB_ACTOR")
if github_user is None:
    print("Error: GITHUB_ACTOR not found.")
    sys.exit(1)

team_name = github_user

# =========================
# Load CSVs
# =========================
submission = pd.read_csv(submission_csv)
labels = pd.read_csv(labels_csv)

# =========================
# Normalize columns
# =========================
labels = labels.rename(columns={"label": "true_label"})
submission = submission.rename(columns={"label": "pred_label"})

# Normalize IDs (VERY IMPORTANT)
labels["id"] = labels["id"].astype(str).str.strip()
submission["id"] = submission["id"].astype(str).str.strip()

# =========================
# Set index for alignment
# =========================
labels = labels.set_index("id")
submission = submission.set_index("id")

# =========================
# Mismatch diagnostics
# =========================
missing_in_submission = labels.index.difference(submission.index)
extra_in_submission = submission.index.difference(labels.index)

print("========== DATA CHECK ==========")
print(f"Total ground truth: {len(labels)}")
print(f"Total predictions: {len(submission)}")
print(f"Missing predictions: {len(missing_in_submission)}")
print(f"Extra predictions: {len(extra_in_submission)}")

if len(missing_in_submission) > 0:
    print("Example missing IDs:", list(missing_in_submission)[:5])

if len(extra_in_submission) > 0:
    print("Example extra IDs:", list(extra_in_submission)[:5])

# =========================
# Align submission to labels
# =========================
submission = submission.reindex(labels.index)

# =========================
# Handle missing predictions
# =========================
if submission["pred_label"].isna().any():
    print("⚠️ Missing predictions detected")

    # OPTION 1: STRICT (recommended)
    # print("❌ Submission rejected: missing predictions")
    # sys.exit(1)

    # OPTION 2: Penalize (used here)
    submission["pred_label"] = submission["pred_label"].fillna(-1)

# =========================
# Final evaluation (FULL dataset)
# =========================
y_true = labels["true_label"]
y_pred = submission["pred_label"]

accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
f1 = round(f1_score(y_true, y_pred, average="macro") * 100, 2)

print("\n========== RESULTS ==========")
print(f"Accuracy for {team_name}: {accuracy:.2f}%")
print(f"F1 Score for {team_name}: {f1:.2f}%")

# =========================
# Leaderboard file
# =========================
leaderboard_file = "final_leaderboard.csv"

new_entry = {
    'Team': team_name,
    'Accuracy': accuracy,
    'F1-Score': f1
}

# =========================
# Load or create leaderboard
# =========================
if os.path.exists(leaderboard_file):
    leaderboard = pd.read_csv(leaderboard_file)

    # 🔒 Only one submission allowed
    if team_name in leaderboard['Team'].values:
        print(f"User '{team_name}' already submitted. Only first submission allowed.")
        sys.exit(1)
else:
    leaderboard = pd.DataFrame(columns=['Team', 'Accuracy', 'F1-Score'])

# =========================
# Add entry
# =========================
leaderboard = pd.concat([leaderboard, pd.DataFrame([new_entry])], ignore_index=True)

# =========================
# Sort leaderboard
# =========================
if metric not in leaderboard.columns:
    print(f"Error: Metric '{metric}' not found.")
    sys.exit(1)

leaderboard = leaderboard.sort_values(by=metric, ascending=False)

# =========================
# Save leaderboard
# =========================
leaderboard.to_csv(leaderboard_file, index=False)

print(f"\n🏆 Leaderboard sorted by {metric}")
print(leaderboard.to_string(index=False))
