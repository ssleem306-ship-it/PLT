import pandas as pd

# 🔧 CHOOSE METRIC HERE: "accuracy" or "f1"
METRIC = "accuracy"   # change to "f1" when needed

# Load CSV
df = pd.read_csv("final_leaderboard.csv")

# Normalize column names
if "F1-score" in df.columns:
    df.rename(columns={"F1-score": "F1"}, inplace=True)

if "F1-Score" in df.columns:
    df.rename(columns={"F1-Score": "F1"}, inplace=True)

# Validate
assert "Accuracy" in df.columns
assert "F1" in df.columns, f"Missing F1 column. Found: {df.columns}"
# assert "F1" in df.columns

# Select metric column
metric_col = "Accuracy" if METRIC.lower() == "accuracy" else "F1"

# Sort by chosen metric
df = df.sort_values(by=metric_col, ascending=False).reset_index(drop=True)

# Ranking (Kaggle-style)
ranks = []
prev_score = None
prev_rank = 0
count = 0

for score in df[metric_col]:
    count += 1
    if score == prev_score:
        rank = prev_rank
    else:
        rank = count
    ranks.append(rank)
    prev_score = score
    prev_rank = rank

df["Rank"] = ranks

# Medal logic
def medal_by_rank(rank):
    if rank == 1:
        return "gold", "🥇"
    elif rank == 2:
        return "silver", "🥈"
    elif rank == 3:
        return "bronze", "🥉"
    else:
        return "", ""

df[["Class", "Medal"]] = df["Rank"].apply(lambda r: pd.Series(medal_by_rank(r)))

# Stats
stats = {
    "participants": len(df),

    "acc_max": df["Accuracy"].max(),
    "acc_min": df["Accuracy"].min(),
    "acc_mean": df["Accuracy"].mean(),
    "acc_median": df["Accuracy"].median(),
    "acc_std": df["Accuracy"].std(),

    "f1_max": df["F1"].max(),
    "f1_min": df["F1"].min(),
    "f1_mean": df["F1"].mean(),
    "f1_median": df["F1"].median(),
    "f1_std": df["F1"].std()
}

# HTML
html_content = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Leaderboard</title>
<style>
body {{
    font-family: Arial;
    background: #f7fbff;
    color: black;
    text-align: center;
    padding: 30px;
}}

table {{
    border-collapse: collapse;
    width: 70%;
    margin: auto;
    background: white;
}}

th, td {{
    padding: 12px;
    border: 1px solid #ddd;
}}

th {{
    background: #eef5fb;
}}

tr:nth-child(even) {{
    background: #f4f7fa;
}}

.gold {{ background-color: #FFD700; font-weight: bold; }}
.silver {{ background-color: #C0C0C0; font-weight: bold; }}
.bronze {{ background-color: #CD7F32; font-weight: bold; }}

.stats {{
    margin-top: 40px;
}}
</style>
</head>

<body>

<h1>🏆 Leaderboard (Ranking by {metric_col})</h1>

<table>
<tr>
<th>Rank</th>
<th>Team</th>
<th>Accuracy</th>
<th>F1-score</th>
</tr>
"""

for _, row in df.iterrows():
    rank_display = f"{row['Medal']} {row['Rank']}" if row['Medal'] else row['Rank']

    html_content += f"""
    <tr class="{row['Class']}">
        <td>{rank_display}</td>
        <td>{row['Team']}</td>
        <td>{row['Accuracy']:.2f}</td>
        <td>{row['F1']:.4f}</td>
    </tr>
    """

html_content += f"""
</table>

<div class="stats">
<h2>📊 Statistics</h2>

<p><b>Participants:</b> {stats['participants']}</p>

<p><b>Accuracy:</b> Max {stats['acc_max']:.2f} | Min {stats['acc_min']:.2f} | 
Mean {stats['acc_mean']:.2f} | Median {stats['acc_median']:.2f} | Std {stats['acc_std']:.2f}</p>

<p><b>F1-score:</b> Max {stats['f1_max']:.4f} | Min {stats['f1_min']:.4f} | 
Mean {stats['f1_mean']:.4f} | Median {stats['f1_median']:.4f} | Std {stats['f1_std']:.4f}</p>

</div>

</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"✅ Leaderboard generated using {metric_col}")
