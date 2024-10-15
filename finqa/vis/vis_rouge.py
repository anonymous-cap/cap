import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("finqa_rougeL.csv")
print(df.head())

fig, ax = plt.subplots(figsize=(12, 6))

# Define a larger bar width
bar_width = 0.15  # increased width of each bar
offset = bar_width /2  # small offset for spacing within each model

x = np.arange(len(df['Model']))

# Define colors for train, val, and test
train_color = '#1f77b4'  # Blue for train
val_color = '#ff7f0e'    # Orange for validation
test_color = '#2ca02c'   # Green for test
# Plotting the bars for each metric with increased width
ax.bar(x - bar_width - offset, df['Train_Original'], width=bar_width, label='Train Original', color=train_color, alpha=0.5)
ax.bar(x - offset, df['Train_Modified'], width=bar_width, label='Train Modified', color=train_color, alpha=0.2)

ax.bar(x + offset, df['Val_Original'], width=bar_width, label='Val Original', color=val_color,alpha=0.5)
ax.bar(x + bar_width + offset, df['Val_Modified'], width=bar_width, label='Val Modified', color=val_color, alpha=0.2)

ax.bar(x + 2*bar_width + offset, df['Test_Original'], width=bar_width, label='Test Original', color=test_color,alpha=0.5)
ax.bar(x + 3*bar_width + offset, df['Test_Modified'], width=bar_width, label='Test Modified', color=test_color, alpha=0.2)

# Customizing the plot
ax.set_ylabel('ROUGE-L',fontsize=16)
ax.set_title('ROUGE-L Scores for Train, Validation, and Test Sets Before and After Text Modification',fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(df['Model'],fontsize=16)
ax.legend(fontsize=16)
# Adding grid lines (only horizontal)
ax.yaxis.grid(True, linestyle='--', linewidth=0.5)  # Add horizontal grid lines

# Display the plot
plt.tight_layout()
plt.savefig('./draws/finqa_rougeL.png', dpi=700)  # Save with high resolution (300 DPI)
plt.show()