import pandas as pd
import sys
from sklearn.metrics import accuracy_score, f1_score
import os

# =========================
# CONFIGURATION
# =========================
# Choose ranking metric: "Accuracy" or "F1-Score"
metric = "Accuracy"   # change to "F1-Score" if needed

# =========================
# Arguments
# =========================
if len(sys.argv) < 3:
    print("Usage: python evaluate.py <submission_csv> <labels_csv>")
    sys.exit(1)

submission_csv = sys.argv[1]
labels_csv = sys.argv[2]

# =========================
# Get team name (GitHub username)
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
# Merge and evaluate
# =========================
labels = labels.rename(columns={"label": "true_label"})
submission = submission.rename(columns={"label": "pred_label"})

merged = pd.merge(labels, submission, on="id")

y_true = merged["true_label"]
y_pred = merged["pred_label"]

accuracy = round(accuracy_score(y_true, y_pred) * 100, 2)
f1 = round(f1_score(y_true, y_pred, average="macro") * 100, 2)

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

    # 🔒 STRICT: only one submission allowed
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
# Sort by selected metric
# =========================
if metric not in leaderboard.columns:
    print(f"Error: Metric '{metric}' not found in leaderboard columns.")
    sys.exit(1)

leaderboard = leaderboard.sort_values(by=metric, ascending=False)

# =========================
# Save leaderboard
# =========================
leaderboard.to_csv(leaderboard_file, index=False)

print(f"\n🏆 Leaderboard sorted by {metric}")
print(leaderboard.to_string(index=False))
