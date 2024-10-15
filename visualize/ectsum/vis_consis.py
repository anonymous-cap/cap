import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("ectsum_95.csv")
print(df.head())


# Define the bar width and positions
bar_width = 0.3
index = np.arange(len(df))

# Create the plot
plt.figure(figsize=(10, 6))
plt.barh(index, df['Drop_Train_Test'], height=bar_width, label='Drop_Train_Test', align='center',color="#5DADE2", alpha=0.6,linestyle='-')
plt.barh(index + bar_width, df['Drop_Train_Val'], height=bar_width, label='Drop_Train_Val', align='center',color="#48C9B0", alpha=0.6)

# Set the y-axis ticks and labels
plt.yticks(index + bar_width / 2, df['Model'],fontsize=21)
plt.xticks(fontsize=21)
# Add labels and title
plt.title('PCR Drop Between Train and Val/Test Sets (ECTSum)', fontsize=21)
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1],fontsize=21)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.xlim(-0.059, 0.157)
plt.xticks(np.arange(-0.05, 0.16, 0.05),fontsize=21)
plt.tight_layout()
plt.savefig('ectsum_consistency.png', dpi=700)  # Save with high resolution (300 DPI)
plt.show()