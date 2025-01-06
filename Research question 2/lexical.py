import pandas as pd
import spacy
import re

nlp = spacy.load("en_core_web_sm")

def lexical_analysis(text):
    doc = nlp(text)
    analysis = {
        "tokens": [token.text for token in doc],
        "pos": [token.pos_ for token in doc],
        "ner": [(ent.text, ent.label_) for ent in doc.ents],
        "lemmas": [token.lemma_ for token in doc]
    }
    return analysis

def extract_high_attribution_tokens(phrases):
    try:
        matches = re.findall(r"\(np\.int32\(1\),\s*'([^']*)'\)", phrases)
        return " ".join(matches)
    except Exception as e:
        return ""

df = pd.read_csv("Research question 2/results/kmeans_clustering_analysis.csv")

df["high_attrib_prompt_tokens"] = df["prompt_phrases"].apply(extract_high_attribution_tokens)
df["high_attrib_continuation_tokens"] = df["continuation_phrases"].apply(extract_high_attribution_tokens)

df["high_attrib_prompt_analysis"] = df["high_attrib_prompt_tokens"].apply(lexical_analysis)
df["high_attrib_continuation_analysis"] = df["high_attrib_continuation_tokens"].apply(lexical_analysis)

df.to_csv("high_attribution_lexical_analysis_results.csv", index=False)

# Extracting tokens into a separate DataFrame
tokens_data = []
for index, row in df.iterrows():
    for token in row["high_attrib_prompt_analysis"]["tokens"]:
        tokens_data.append({"type": "prompt", "token": token})
    for token in row["high_attrib_continuation_analysis"]["tokens"]:
        tokens_data.append({"type": "continuation", "token": token})

tokens_df = pd.DataFrame(tokens_data)
tokens_df.to_csv("high_attribution_tokens.csv", index=False)

print("Lexical analysis results for high-attribution tokens and separate token CSV saved.")
