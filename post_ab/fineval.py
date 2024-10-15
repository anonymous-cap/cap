import pandas as pd
import numpy as np
data = {
    "Model": [
        "LLaMA-8B", "Mistral-7B", "FinMA-Full-7B", "FinMA-NLP-7B",
        "Baichuan-13B", "Disc-Fin-13B", "GPT-4o-mini",
        "LLaMA-8B", "Mistral-7B", "FinMA-Full-7B", "FinMA-NLP-7B",
        "Baichuan-13B", "Disc-Fin-13B", "GPT-4o-mini"
    ],
    "Set": ["Dev","Dev", "Dev","Dev","Dev","Dev", "Dev",
             "Test", "Test",  "Test",  "Test",  "Test", "Test", "Test"],
    "EM_Ori": [
        0, 0, 0.1824, 0.1176,
        0.4235, 0.4647, 0,
        0, 0, 0.1706, 0.1095,
        0.4544, 0.4327, 0
    ],
    "Consistency": [
        0.6059, 0.5294, 0.2706, 0.2765,
        0.4765, 0.4824, 0.7706,
        0.5308, 0.4944, 0.2502, 0.2407,
        0.4674, 0.4083, 0.7489
    ]
}

df = pd.DataFrame(data)
def tanh_transform(x, y, alpha):
    return np.tanh((x + alpha) / (y+alpha))


def linear_transform(x, y, alpha):
    return (x+alpha)/(y+alpha)
def sigmoid_transform(x, y, alpha):
    return 1 / (1 + np.exp(-(x + alpha) / (y+alpha)))

alphas = [0, 0.001, 0.01, 0.1]
# Calculate new columns for each alpha
for alpha in alphas:
    df[f'TANH, alpha={alpha}'] = tanh_transform(df['EM_Ori'], df['Consistency'], alpha)
    df[f'Linear, alpha={alpha}']=linear_transform(df['EM_Ori'], df['Consistency'], alpha)
    df[f'Sigmoid, alpha={alpha}'] = sigmoid_transform(df['EM_Ori'], df['Consistency'], alpha)
df.to_csv('Fineval.csv', index=False)
