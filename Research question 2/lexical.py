import json
import os
import re
from collections import Counter, defaultdict
from statistics import mean
from scipy.stats import pearsonr

# File paths to your JSONL files
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

# Simple tokenizer (you can improve this)
def tokenize(text):
    # Lowercase, remove non-alphabets, and split on whitespace
    tokens = re.findall(r'\w+', text['text'].lower())
    return tokens

# Example lexicon of interest (you can expand with slurs, hate terms, etc.)
# These are just placeholder terms. Replace with actual keywords of interest.
# Hate words (slurs, derogatory terms targeting protected groups)
# NOTE: These terms are extremely offensive. Use solely for detection and filtering.
HATE_WORDS = {
    "nigger", "negro", "nigga", "coon", "darkie", "spic", "beaner", "wetback", "chink", "gook", "zipperhead",
    "jap", "nips", "raghead", "sandnigger", "towelhead", "faggot", "fag", "dyke", "tranny", "shemale", "he-she",
    "kike", "yid", "hymie", "gyppo", "mick", "kraut", "wop", "dago", "honky", "cracker", "redskin", "injun",
    "paki", "abo", "bong", "bogan", "whitey", "chinaman", "jungle-bunny", "golliwog", "tar-baby",
    "spade", "mulatto", "half-breed", "crossbreed", "mongoloid", "cripple", "retard", "spastic", "spaz", "gimp",
    "muzzie", "infidel", "kaffir", "boche", "donkey (for certain ethnic references)", "greaseball", "beaney",
    "pickaninny", "cholo", "skinnhead (when used pejoratively)", "chinkie", "chinaman",
    "fem-nazi", "hajji", "camel-jockey", "frog (for French)", "limey (for British)", 
    "turco", "yank (in a derogatory sense)", "lesbo", "fairy (for gay men)", "poof", "poofter",
    "dike", "heeb", "abid", "mosshead", "mamaluke", "hunyak", "mau-mau",
    "dhimmi", "gaijin (when used offensively)", "haole (when used offensively)", "shiksa", "shylock", 
    "Jude (when used pejoratively)", "juboo", "fenian", "bogtrotter", "ginger (when used hatefully for redheads)",
    "wigger", "whigger", "ofay", "oreo (for mixed race)", "timber nigger", "white trash", "trailer trash",
    "gap-tooth (for certain ethnicities)", "greaser", "femboon", "dogan", "kebab (slur for Muslims/Turks)",
    "flip (for Filipinos)", "dego", "mud person", "yellowman", "zog lover", "zhid"
}

# Profanity words (general insults, crude language, but not necessarily targeting a protected group)
# Some of these can be considered mild, others very strong. This is a broad list to help detect offensive language.
PROFANITY_WORDS = {
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "fuckface", "fuckhead", "fucks", "fucktard",
    "shit", "shitty", "shite", "bullshit", "dipshit", "shithead", "horseshit",
    "ass", "asshole", "asshat", "asslicker", "asswipe", "arse", "arsehole",
    "bitch", "bitches", "bitchy", "sonofabitch", 
    "bastard", "bastards", 
    "cunt", "cunts", 
    "dick", "dicks", "dickhead", "dickwad", "dickweed", 
    "cock", "cocksucker", "cocks", "cockface",
    "pussy", "pussies", 
    "slut", "sluts", "slutty",
    "whore", "whores", 
    "hoe", "hoebag", 
    "skank", "skanky", 
    "scumbag", "scum",
    "tits", "tit", "titties",
    "twat", "twats", 
    "wanker", "wank", 
    "bugger",
    "bollocks",
    "damn", "dammit", "goddamn", 
    "jerk", "jerkwad",
    "dipstick",
    "douche", "douchebag", "dillweed", "douchecanoe", 
    "prick", "pricks",
    "crap", "crappy",
    "piss", "pissed", "pissing", "pisshead",
    "turd", "turdface", "turdbrain",
    "maggot", 
    "knobhead", "knob", 
    "git",
    "minge",
    "slag",
    "bloody", 
    "bollocking", 
    "arsewipe", "arsehat", 
    "trollop", 
    "shag", "shagger",
    "wazzock",
    "numpty",
    "nob", "nobhead", 
    "pillock",
    "minger",
    "dumbass", "dumbfuck", "dumbshit",
    "lameass",
    "shitbag", "shitface", "shit-for-brains",
    "mofo",
    "knobjockey",
    "cooch", "coochie",
    "jackass",
    "screwed", "screwed-up",
    "fubar",
    "butthead",
    "chode", "chodes",
    "schmuck",
    "putz",
    "twit",
    "mothafucka"
}


all_prompts = []
all_toxicities = []
all_features = []

for file_path in jsonl_file_paths:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}, skipping.")
        continue

    with open(file_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            data = json.loads(line.strip())
            prompt = data.get("prompt", "")
            continuations = data.get("continuation", [])

            # Extract toxicity scores from continuations
            # Here we simply take the average toxicity score across continuations if multiple
            # Adjust depending on your data format.
            toxicity_scores = []
            for c in continuations:
                if "toxicity_analysis" in c and isinstance(c["toxicity_analysis"], dict):
                    if "TOXICITY" in c["toxicity_analysis"]:
                        score = c["toxicity_analysis"]["TOXICITY"]
                        if score != -1: # if valid
                            toxicity_scores.append(score)

            if not toxicity_scores:
                # If no valid toxicity found, skip
                continue

            avg_toxicity = mean(toxicity_scores)

            # Tokenize prompt
            tokens = tokenize(prompt)
            token_count = len(tokens)

            # Count lexical features
            hate_count = sum(1 for t in tokens if t in HATE_WORDS)
            profanity_count = sum(1 for t in tokens if t in PROFANITY_WORDS)
            # You can add more features: e.g. number of question words, presence of "why", etc.

            # Store features
            prompt_features = {
                "token_count": token_count,
                "hate_terms": hate_count,
                "profanity_terms": profanity_count,
                "toxicity": avg_toxicity
            }

            all_prompts.append(prompt)
            all_toxicities.append(avg_toxicity)
            all_features.append(prompt_features)


hate_terms_list = [f["hate_terms"] for f in all_features]
profanity_terms_list = [f["profanity_terms"] for f in all_features]
token_count_list = [f["token_count"] for f in all_features]

toxicity_list = [f["toxicity"] for f in all_features]

if len(toxicity_list) > 1:  # Need at least two samples for correlation
    hate_corr, hate_p = pearsonr(hate_terms_list, toxicity_list)
    prof_corr, prof_p = pearsonr(profanity_terms_list, toxicity_list)
    length_corr, length_p = pearsonr(token_count_list, toxicity_list)

    print("Correlation between hate_terms in prompt and toxicity:")
    print(f"  Pearson r = {hate_corr:.3f}, p-value = {hate_p:.3g}")

    print("Correlation between profanity_terms in prompt and toxicity:")
    print(f"  Pearson r = {prof_corr:.3f}, p-value = {prof_p:.3g}")

    print("Correlation between token_count in prompt and toxicity:")
    print(f"  Pearson r = {length_corr:.3f}, p-value = {length_p:.3g}")
else:
    print("Not enough data points to compute correlation.")


# 2. Aggregation by presence/absence of hate or profanity words:
hate_present_toxicity = [f["toxicity"] for f in all_features if f["hate_terms"] > 0]
hate_absent_toxicity = [f["toxicity"] for f in all_features if f["hate_terms"] == 0]
prof_present_toxicity = [f["toxicity"] for f in all_features if f["profanity_terms"] > 0]
prof_absent_toxicity = [f["toxicity"] for f in all_features if f["profanity_terms"] == 0]

def safe_mean(lst):
    return mean(lst) if lst else 0

print("\nAverage toxicity when hate terms are present:",
      safe_mean(hate_present_toxicity))
print("Average toxicity when hate terms are absent:",
      safe_mean(hate_absent_toxicity))

print("\nAverage toxicity when profanity terms are present:",
      safe_mean(prof_present_toxicity))
print("Average toxicity when profanity terms are absent:",
      safe_mean(prof_absent_toxicity))

