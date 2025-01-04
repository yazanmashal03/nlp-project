import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Research question 2/results/high_attribution_lexical_analysis_results.csv")

def count_pos(pos_analysis):
    try:
        pos_data = eval(pos_analysis)
        pos_tags = pos_data.get("pos", [])
        pos_count = pd.Series(pos_tags).value_counts().to_dict()
        return pos_count
    except Exception as e:
        return {}

df["prompt_pos_counts"] = df["high_attrib_prompt_analysis"].apply(count_pos)
df["continuation_pos_counts"] = df["high_attrib_continuation_analysis"].apply(count_pos)

total_prompt_pos = pd.Series([pos for pos_count in df["prompt_pos_counts"] for pos in pos_count.keys()]).value_counts()
total_continuation_pos = pd.Series([pos for pos_count in df["continuation_pos_counts"] for pos in pos_count.keys()]).value_counts()

total_prompt_sum = total_prompt_pos.sum()
total_continuation_sum = total_continuation_pos.sum()
prompt_percentages = (total_prompt_pos / total_prompt_sum) * 100
continuation_percentages = (total_continuation_pos / total_continuation_sum) * 100

pos_comparison = pd.DataFrame({
    "Prompt POS Count": total_prompt_pos,
    "Prompt POS %": prompt_percentages,
    "Continuation POS Count": total_continuation_pos,
    "Continuation POS %": continuation_percentages
})

pos_comparison[["Prompt POS Count", "Continuation POS Count"]].plot(kind='bar', figsize=(12, 6))
plt.title("POS Tag Distribution for High-Attribution Tokens")
plt.xlabel("Part of Speech")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(pos_comparison)

pos_comparison.to_csv("pos_tag_counts_with_percentages_high_attribution.csv")
