import pandas as pd
import spacy
import re

# Load the English language model
nlp = spacy.load("en_core_web_sm")

def dependency_analysis(text):
    doc = nlp(text)
    analysis = {
        "tokens": [token.text for token in doc],
        "dep_info": [(token.text, token.dep_, token.head.text) for token in doc],
    }
    return analysis

def extract_high_attribution_tokens(phrases):
    try:
        matches = re.findall(r"\(np\.int32\(1\),\s*'([^']*)'\)", phrases)
        return " ".join(matches) 
    except Exception:
        return ""

if __name__ == "__main__":
    df = pd.read_csv("Research question 2/results/kmeans_clustering_analysis.csv")

    df["high_attrib_prompt_tokens"] = df["prompt_phrases"].apply(extract_high_attribution_tokens)
    df["high_attrib_continuation_tokens"] = df["continuation_phrases"].apply(extract_high_attribution_tokens)

    df["high_attrib_prompt_analysis"] = df["high_attrib_prompt_tokens"].apply(dependency_analysis)
    df["high_attrib_continuation_analysis"] = df["high_attrib_continuation_tokens"].apply(dependency_analysis)

    df["high_attrib_prompt_analysis"] = df["high_attrib_prompt_analysis"].apply(lambda x: str(x))
    df["high_attrib_continuation_analysis"] = df["high_attrib_continuation_analysis"].apply(lambda x: str(x))

    df.to_csv("high_attribution_dependency_analysis_results.csv", index=False)

    print("Dependency analysis results for high-attribution tokens saved to 'high_attribution_dependency_analysis_results.csv'.")
