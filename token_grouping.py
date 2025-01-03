import os
import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Set the directory containing JSON files
directory = "processed_results"

# Define the number of clusters for K-Means clustering
N_CLUSTERS = 2

# Initialize results list
results = []

def perform_kmeans_clustering(attributions, n_clusters):
    """Perform K-Means clustering on attribution scores."""
    if len(attributions) < n_clusters:
        # If the number of attributions is less than the number of clusters, assign all to one cluster
        return [0] * len(attributions)
    
    X = np.array(attributions).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    return labels

def group_high_attribution(tokens, attributions, n_clusters):
    """Group tokens based on K-Means clustering of attributions."""
    if len(tokens) == 0 or len(attributions) == 0:
        return []
    
    # Normalize attributions to handle varying scales
    max_abs_attr = max(abs(attr) for attr in attributions)
    normalized_attributions = (
        np.array(attributions) / max_abs_attr if max_abs_attr != 0 else np.array(attributions)
    )

    # Perform K-Means clustering
    labels = perform_kmeans_clustering(normalized_attributions, n_clusters)

    # Reconstruct phrases based on cluster labels
    phrases = []
    current_label = labels[0]
    current_phrase = [tokens[0]]

    for token, label in zip(tokens[1:], labels[1:]):
        if label == current_label:
            current_phrase.append(token)
        else:
            phrases.append((current_label, " ".join(current_phrase)))
            current_phrase = [token]
            current_label = label

    # Append the last phrase
    phrases.append((current_label, " ".join(current_phrase)))

    return phrases

def do_token_grouping():
    # Process each JSON file
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = []
                for line in file:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)

                # Process prompt and continuation for each entry
                for entry in data:
                    prompt = entry.get("prompt", {})
                    continuation = entry.get("continuation", [])

                    # Extract high-attribution phrases from the prompt
                    prompt_phrases = group_high_attribution(
                        tokens=prompt.get("tokens", []),
                        attributions=prompt.get("attributions", []),
                        n_clusters=N_CLUSTERS
                    )

                    # Extract high-attribution phrases from each continuation
                    for cont in continuation:
                        cont_phrases = group_high_attribution(
                            tokens=cont.get("tokens", []),
                            attributions=cont.get("attributions", []),
                            n_clusters=N_CLUSTERS
                        )

                        # Collect results
                        results.append({
                            "filename": filename,
                            "prompt_text": prompt.get("text", ""),
                            "prompt_phrases": prompt_phrases,
                            "continuation_text": cont.get("text", ""),
                            "continuation_phrases": cont_phrases,
                        })

do_token_grouping()

# Convert results into a DataFrame for analysis
df = pd.DataFrame(results)

# Save results for inspection
df.to_csv("kmeans_clustering_analysis.csv", index=False)

# Display the first rows for validation
print(df.head())
