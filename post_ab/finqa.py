import pandas as pd
import numpy as np

# Creating the data based on the extracted text
data = {
    "Model": [
        "LLaMA", "Mistral", "FinMA-Full", "FinMA-NLP",
        "Baichuan", "Disc-Fin", "GPT-4o-mini",
        "LLaMA", "Mistral", "FinMA-Full", "FinMA-NLP",
        "Baichuan", "Disc-Fin", "GPT-4o-mini",
        "LLaMA", "Mistral", "FinMA-Full", "FinMA-NLP",
        "Baichuan", "Disc-Fin", "GPT-4o-mini"
    ],
    "Ori_rouge": [
        0.0021, 0.0209, 0.2304, 0.1481,
        0.0084, 0.0097, 0.0209,
        0.0021, 0.0229, 0.0943, 0.092,
        0.01, 0.0118, 0.0217,
        0.0022, 0.0201, 0.1084, 0.106,
        0.0081, 0.0144, 0.0208
    ],
    "Consistency": [
        0.8326, 0.5554, 0.727, 0.6993,
        0.2506, 0.512, 0.4562,
        0.83, 0.5679, 0.5662, 0.638,
        0.2546, 0.5554, 0.4604,
        0.8246, 0.5656, 0.5706, 0.6219,
        0.2466, 0.4796, 0.4505
    ]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Transformation functions
def tanh_transform(x, y, alpha):
    return np.tanh((x + alpha) / (y + alpha))

def linear_transform(x, y, alpha):
    return (x + alpha) / (y + alpha)

def sigmoid_transform(x, y, alpha):
    return 1 / (1 + np.exp(-(x + alpha) / (y + alpha)))

alphas = [0, 0.001, 0.01, 0.1]
# Calculate new columns for each alpha
for alpha in alphas:
    df[f'TANH, alpha={alpha}'] = tanh_transform(df['Ori_rouge'], df['Consistency'], alpha)
    df[f'Linear, alpha={alpha}'] = linear_transform(df['Ori_rouge'], df['Consistency'], alpha)
    df[f'Sigmoid, alpha={alpha}'] = sigmoid_transform(df['Ori_rouge'], df['Consistency'], alpha)

# Save the modified DataFrame to CSV
df.to_csv('FinQA.csv', index=False)
