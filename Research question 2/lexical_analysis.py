import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset containing high-attribution lexical analysis results
df = pd.read_csv("Research question 2/results/high_attribution_lexical_analysis_results.csv")

# Function to count POS tags in the analyzed data
def count_pos(pos_analysis):
    try:
        # Convert the string back into a Python dictionary
        pos_data = eval(pos_analysis)
        pos_tags = pos_data.get("pos", [])
        # Count the occurrences of each POS tag
        pos_count = pd.Series(pos_tags).value_counts().to_dict()
        return pos_count
    except Exception as e:
        return {}

# Count POS tags in both prompts and continuations
df["prompt_pos_counts"] = df["high_attrib_prompt_analysis"].apply(count_pos)
df["continuation_pos_counts"] = df["high_attrib_continuation_analysis"].apply(count_pos)

# Combine results into a single count for a better overview
total_prompt_pos = pd.Series([pos for pos_count in df["prompt_pos_counts"] for pos in pos_count.keys()]).value_counts()
total_continuation_pos = pd.Series([pos for pos_count in df["continuation_pos_counts"] for pos in pos_count.keys()]).value_counts()

# Calculate percentages
total_prompt_sum = total_prompt_pos.sum()
total_continuation_sum = total_continuation_pos.sum()
prompt_percentages = (total_prompt_pos / total_prompt_sum) * 100
continuation_percentages = (total_continuation_pos / total_continuation_sum) * 100

# Create a combined DataFrame with counts and percentages
pos_comparison = pd.DataFrame({
    "Prompt POS Count": total_prompt_pos,
    "Prompt POS %": prompt_percentages,
    "Continuation POS Count": total_continuation_pos,
    "Continuation POS %": continuation_percentages
})

# Plotting the POS counts as a bar chart
pos_comparison[["Prompt POS Count", "Continuation POS Count"]].plot(kind='bar', figsize=(12, 6))
plt.title("POS Tag Distribution for High-Attribution Tokens")
plt.xlabel("Part of Speech")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display percentages for each POS tag
print(pos_comparison)

# Optional: Save the results to a CSV file
pos_comparison.to_csv("pos_tag_counts_with_percentages_high_attribution.csv")
