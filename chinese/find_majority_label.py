import json
from collections import Counter

# Load the dev.json file
with open("./data/dev.json", "r") as f:
    data = json.load(f)

# Collect all scores from every "choices" list
all_scores = []
for key, entry in data.items():
    all_scores.extend(entry["choices"])

# Count frequency of each score
counter = Counter(all_scores)

# Find the most common score
majority_label, count = counter.most_common(1)[0]

print(f"Majority label: {majority_label} (appears {count} times)")
