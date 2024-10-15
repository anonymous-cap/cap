import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("fineval_consis.csv")
print(df.head())


# Define the bar width and positions
bar_width = 0.3
index = np.arange(len(df))

# Create the plot
plt.figure(figsize=(10, 6))
plt.barh(index, df['Drop'], height=bar_width, label='Drop', align='center',color="darkseagreen", alpha=0.7,linestyle='-')

# Set the y-axis ticks and labels
plt.yticks(index + bar_width / 2, df['Model'],fontsize=21)

plt.xlim(-0.1, 0.1)
plt.xticks(np.arange(-0.1, 0.13, 0.05),fontsize=21)
# Add labels and title
plt.title('PCR Drop Between Dev and Val Sets (FinEval)', fontsize=21)
plt.legend(fontsize=21)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.savefig('fineval_consistency_v2.png', dpi=700)  # Save with high resolution (300 DPI)
plt.show()