import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Research question 3/results/high_attribution_dependency_analysis_results.csv")

def count_dependency_labels(dep_analysis):
    """
    Converts the stored string of dict data into a dictionary,
    extracts the 'dep_info' list, and counts the occurrences 
    of each dependency label.
    """
    try:
        dep_data = eval(dep_analysis)

        dep_info = dep_data.get("dep_info", [])
        
        dep_labels = [tup[1] for tup in dep_info]
        
        dep_count = pd.Series(dep_labels).value_counts().to_dict()
        
        return dep_count
    except Exception:
        return {}

df["prompt_dep_counts"] = df["high_attrib_prompt_analysis"].apply(count_dependency_labels)
df["continuation_dep_counts"] = df["high_attrib_continuation_analysis"].apply(count_dependency_labels)

total_prompt_dep = pd.Series([
    label 
    for dep_dict in df["prompt_dep_counts"] 
    for label in dep_dict.keys()
]).value_counts()

total_continuation_dep = pd.Series([
    label 
    for dep_dict in df["continuation_dep_counts"] 
    for label in dep_dict.keys()
]).value_counts()

total_prompt_sum = total_prompt_dep.sum()
total_continuation_sum = total_continuation_dep.sum()

prompt_percentages = (total_prompt_dep / total_prompt_sum) * 100
continuation_percentages = (total_continuation_dep / total_continuation_sum) * 100

dep_comparison = pd.DataFrame({
    "Prompt Dep Count": total_prompt_dep,
    "Prompt Dep %": prompt_percentages,
    "Continuation Dep Count": total_continuation_dep,
    "Continuation Dep %": continuation_percentages
})

dep_comparison[["Prompt Dep Count", "Continuation Dep Count"]].plot(
    kind='bar', 
    figsize=(12, 6)
)
plt.title("Dependency Label Distribution for High-Attribution Tokens")
plt.xlabel("Dependency Label")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(dep_comparison)

dep_comparison.to_csv("dependency_label_counts_with_percentages.csv")
