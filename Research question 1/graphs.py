import pandas as pd
import matplotlib.pyplot as plt

# Data from the provided table
data = {
    "Model": [
        "Gemma 7B", "Gemma 7B", "Bloom 7B", "Bloom 7B", 
        "LLaMa 3.1 8B", "LLaMa 3.1 8B", "LLaMa 3.1 8B Instruct", "LLaMa 3.1 8B Instruct", 
        "Mistral 7B", "Mistral 7B", "Mistral 7B Instruct", "Mistral 7B Instruct", 
        "Mistral MoE 8x7B", "Mistral MoE 8x7B"
    ],
    "Dataset Type": [
        "Toxic", "Non-Toxic", "Toxic", "Non-Toxic", 
        "Non-Toxic", "Toxic", "Non-Toxic", "Toxic", 
        "Non-Toxic", "Toxic", "Non-Toxic", "Toxic", 
        "Non-Toxic", "Toxic"
    ],
    "Expected Maximum Toxicity": [
        0.84, 0.35, 0.81, 0.09, 
        0.35, 0.84, 0.22, 0.56, 
        0.35, 0.86, 0.19, 0.66, 
        0.33, 0.84
    ],
    "Toxicity Probability (%)": [
        98.41, 20.92, 95.57, 10.33, 
        20.33, 98.75, 2.58, 62.79, 
        21.67, 99.50, 4.67, 79.26, 
        5.92, 86.96
    ]
}

# Create the dataframe
df = pd.DataFrame(data)

# Separate data by Dataset Type for better visualization
toxic_df = df[df["Dataset Type"] == "Toxic"]
non_toxic_df = df[df["Dataset Type"] == "Non-Toxic"]

# Plotting the data
plt.figure(figsize=(14, 8))

# Toxic Data
plt.bar(toxic_df["Model"], toxic_df["Toxicity Probability (%)"], label="Toxic Inputs", alpha=0.7, color="red")
# Non-Toxic Data
plt.bar(non_toxic_df["Model"], non_toxic_df["Toxicity Probability (%)"], label="Non-Toxic Inputs", alpha=0.7, color="blue")

# Customizing the plot
plt.xticks(rotation=45, ha="right")
plt.ylabel("Toxicity Probability (%)")
plt.title("Toxicity Probability vs Generative Models")
plt.legend()
plt.tight_layout()

# Show the plot
plt.show()

