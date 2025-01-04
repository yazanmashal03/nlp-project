import pandas as pd
import spacy
import re

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------------------------
# 1. Function to perform dependency analysis (instead of lexical analysis)
# ------------------------------------------------------------------
def dependency_analysis(text):
    """
    Returns a structured overview of tokens and their dependency relations.
    The core piece is 'dep_info': a list of (token_text, dependency_label, head_text)
    that shows how each token depends on its head in the parse tree.
    """
    doc = nlp(text)
    analysis = {
        "tokens": [token.text for token in doc],
        "dep_info": [(token.text, token.dep_, token.head.text) for token in doc],
    }
    return analysis

# ------------------------------------------------------------------
# 2. Function to extract high-attribution tokens from the phrases column
#    (kept as in your original code)
# ------------------------------------------------------------------
def extract_high_attribution_tokens(phrases):
    try:
        # Match tokens where the score is explicitly marked as 1
        matches = re.findall(r"\(np\.int32\(1\),\s*'([^']*)'\)", phrases)
        return " ".join(matches)  # Return high-attribution tokens as a string
    except Exception:
        return ""

# ------------------------------------------------------------------
# 3. Main script: load your dataset, extract high-attribution tokens,
#    and run dependency parsing on them
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("Research question 2/results/kmeans_clustering_analysis.csv")

    # Extract high-attribution tokens from both prompts and continuations
    df["high_attrib_prompt_tokens"] = df["prompt_phrases"].apply(extract_high_attribution_tokens)
    df["high_attrib_continuation_tokens"] = df["continuation_phrases"].apply(extract_high_attribution_tokens)

    # Apply dependency analysis only on high-attribution tokens
    df["high_attrib_prompt_analysis"] = df["high_attrib_prompt_tokens"].apply(dependency_analysis)
    df["high_attrib_continuation_analysis"] = df["high_attrib_continuation_tokens"].apply(dependency_analysis)

    # Convert the resulting dictionaries to strings for CSV compatibility
    df["high_attrib_prompt_analysis"] = df["high_attrib_prompt_analysis"].apply(lambda x: str(x))
    df["high_attrib_continuation_analysis"] = df["high_attrib_continuation_analysis"].apply(lambda x: str(x))

    # Save the results to a CSV file
    df.to_csv("high_attribution_dependency_analysis_results.csv", index=False)

    print("Dependency analysis results for high-attribution tokens saved to 'high_attribution_dependency_analysis_results.csv'.")
