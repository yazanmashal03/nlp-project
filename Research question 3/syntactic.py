import json
import os
import re
from statistics import mean
from scipy.stats import pearsonr
import spacy

# Load a spaCy model (ensure it's installed beforehand)
# Run in terminal if not installed: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

jsonl_file_paths = [
    'gemma-7b-output/gemma_7b_toxic_results_processed_analyzed.jsonl',
    'gemma-7b-output/gemma_7b_nontoxic_results_processed_analyzed.jsonl',
    'bloom-7b-output/bloom_7b_toxic_results_processed_analyzed.jsonl',
    'bloom-7b-output/bloom_7b_nontoxic_results_processed_analyzed.jsonl',
    'llama3.1_filtered_output/non_toxic_results_processed_analyzed.jsonl',
    'llama3.1_filtered_output/toxic_results_processed_analyzed.jsonl',
    'llama3.1_instruct_filtered_output/non_toxic_results_processed_analyzed.jsonl',
    'llama3.1_instruct_filtered_output/toxic_results_processed_analyzed.jsonl',
    'mistral-7b-output/mistral_7b_nontoxic_results_processed_analyzed.jsonl',
    'mistral-7b-output/mistral_7b_toxic_results_processed_analyzed.jsonl',
    'mistral-7b-instruct-output/mistral_7b_instruct_nontoxic_results_processed_analyzed.jsonl',
    'mistral-7b-instruct-output/mistral_7b_instruct_toxic_results_processed_analyzed.jsonl',
    'mistral-8x7b-output/moe_nontoxic_results_processed_analyzed.jsonl',
    'mistral-8x7b-output/moe_toxic_results_processed_analyzed.jsonl',
]

all_features = []
all_toxicities = []

for file_path in jsonl_file_paths:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping.")
        continue

    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            prompt = data.get("prompt", "")
            continuations = data.get("continuation", [])

            # Extract toxicity scores
            toxicity_scores = []
            for c in continuations:
                ta = c.get("toxicity_analysis", {})
                if isinstance(ta, dict) and "TOXICITY" in ta and ta["TOXICITY"] != -1:
                    toxicity_scores.append(ta["TOXICITY"])

            if not toxicity_scores:
                continue

            avg_toxicity = mean(toxicity_scores)

            # Process prompt with spaCy to get syntactic features
            doc = nlp(prompt['text'])

            # Example syntactic features:
            # 1. Average sentence length (in tokens)
            sentences = list(doc.sents)
            if len(sentences) > 0:
                avg_sentence_length = mean([len(sent) for sent in sentences])
            else:
                avg_sentence_length = 0

            # 2. Count of different POS tags
            num_verbs = sum(token.pos_ == "VERB" for token in doc)
            num_nouns = sum(token.pos_ == "NOUN" for token in doc)
            num_adjs = sum(token.pos_ == "ADJ" for token in doc)
            num_advs = sum(token.pos_ == "ADV" for token in doc)
            
            # 3. Ratio of subordinate clauses (approximation by counting "mark" dependency)
            # 'mark' dependency often introduces a subordinate clause
            num_sub_clauses = sum(token.dep_ == "mark" for token in doc)
            
            # 4. Presence of interrogative sentences (questions)
            # Check if any sentence ends with '?'
            has_question = any(sent.text.strip().endswith('?') for sent in sentences)
            question_flag = 1 if has_question else 0

            # Store features
            syntactic_features = {
                "avg_sentence_length": avg_sentence_length,
                "num_verbs": num_verbs,
                "num_nouns": num_nouns,
                "num_adjs": num_adjs,
                "num_advs": num_advs,
                "num_sub_clauses": num_sub_clauses,
                "is_question": question_flag,
                "toxicity": avg_toxicity
            }

            all_features.append(syntactic_features)
            all_toxicities.append(avg_toxicity)

# Perform correlation analysis
if len(all_features) > 1:
    feature_names = ["avg_sentence_length", "num_verbs", "num_nouns", "num_adjs", "num_advs", "num_sub_clauses", "is_question"]
    for fn in feature_names:
        feat_values = [f[fn] for f in all_features]
        tox_values = [f["toxicity"] for f in all_features]
        r, p = pearsonr(feat_values, tox_values)
        print(f"Correlation between {fn} and toxicity:")
        print(f"  Pearson r = {r:.3f}, p-value = {p:.3g}")

    # Example aggregation by binary feature (questions)
    questions_present = [f["toxicity"] for f in all_features if f["is_question"] == 1]
    no_questions = [f["toxicity"] for f in all_features if f["is_question"] == 0]

    def safe_mean(lst):
        return mean(lst) if lst else 0

    print("\nAverage toxicity when prompts contain a question:", safe_mean(questions_present))
    print("Average toxicity when prompts do not contain a question:", safe_mean(no_questions))
else:
    print("Not enough data points to compute correlation.")
