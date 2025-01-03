import json
from sklearn.cluster import KMeans
import numpy as np
import chardet

# Filepath to the JSONL file
# FILE_PATH = 'processed_results/mistral_7b_instruct_nontoxic_results_processed_analyzed_highly_toxic.jsonl'
FILE_PATH = 'processed_results/bloom_7b_nontoxic_results_processed_analyzed_highly_toxic.jsonl'
N_CLUSTERS = 2  # e.g., cluster 0: Non-Toxic, cluster 1: Toxic


def detect_file_encoding(file_path):
    """Detect the encoding of a file."""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))  # Read first 10,000 bytes
    return result['encoding']


def load_jsonl(file_path):
    """Load and parse all lines of a JSONL file."""
    try:
        encoding = detect_file_encoding(file_path)
        print(f"Detected file encoding: {encoding}")
        with open(file_path, 'r', encoding=encoding, errors='replace') as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_number}: {e}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except UnicodeDecodeError as e:
        print(f"Unicode decode error: {e}")


def extract_prompt_data(data):
    """Extract prompt text, tokens, and attribution scores."""
    prompt = data.get("prompt", {})
    prompt_text = prompt.get("text", "")
    prompt_tokens = prompt.get("tokens", [])
    prompt_attributions = prompt.get("attributions", [])
    return prompt_text, prompt_tokens, prompt_attributions


def perform_kmeans_clustering(attribution_scores, n_clusters=2):
    """Perform K-Means clustering on attribution scores."""
    X = np.array(attribution_scores).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    return labels


def reconstruct_phrases(tokens, labels):
    """Reconstruct sequences of tokens preserving order and labels."""
    if len(tokens) == 0 or len(labels) == 0 or len(tokens) != len(labels):
        print("Error: Tokens and labels must be of the same length and non-empty.")
        return []

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
    phrases.append((current_label, " ".join(current_phrase)))  # Append the last phrase
    return phrases


def extract_toxic_sequences(phrases, toxic_label=1):
    """Extract sequences labeled as toxic."""
    return [phrase for label, phrase in phrases if label == toxic_label]


def main():
    # Initialize accumulators
    all_prompt_texts = []
    all_toxic_sequences = []

    # Process each JSON object in the JSONL file
    for data in load_jsonl(FILE_PATH):
        if data is None:
            continue

        # Extract prompt data
        prompt_text, prompt_tokens, prompt_attributions = extract_prompt_data(data)
        if prompt_text:
            all_prompt_texts.append(prompt_text)

        # Verify lengths
        if len(prompt_tokens) != len(prompt_attributions):
            print("Warning: The number of prompt tokens does not match the number of attribution scores.")
            continue

        # Perform clustering on prompt attribution scores
        labels = perform_kmeans_clustering(prompt_attributions, N_CLUSTERS)

        # Reconstruct phrases based on cluster labels
        phrases = reconstruct_phrases(prompt_tokens, labels)
        print("\nClustered Prompt Tokens and Their Labels:")
        for label, phrase in phrases:
            print(f"Label {label}: {phrase}")

        # Extract and collect toxic expressions from prompt
        toxic_sequences = extract_toxic_sequences(phrases, toxic_label=1)
        print("\nToxic Expressions in Prompt (Label = 1):", toxic_sequences)
        all_toxic_sequences.extend(toxic_sequences)

    # Optional: After processing all lines, you can perform additional analysis
    print("\n--- Summary ---")
    print(f"Total Prompts Processed: {len(all_prompt_texts)}")
    print(f"Total Toxic Expressions Found: {len(all_toxic_sequences)}")
    # Example: Display all unique toxic expressions
    unique_toxic = set(all_toxic_sequences)
    print(f"Unique Toxic Expressions ({len(unique_toxic)}):")
    for expr in unique_toxic:
        print(f"- {expr}")


if __name__ == "__main__":
    main()