import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("res_consis.csv")
print(df.head())


# Define the bar width and positions
bar_width = 0.3
index = np.arange(len(df))

# Create the plot
plt.figure(figsize=(10, 6))
plt.barh(index, df['Drop'], height=bar_width, label='Drop', align='center',color="#48C9B0", alpha=0.6,linestyle='-')

# Set the y-axis ticks and labels
plt.yticks(index + bar_width / 2, df['Model'],fontsize=21)
plt.xlim(-0.05, 0.157)
plt.xticks(np.arange(-0.05, 0.16, 0.05),fontsize=21)
plt.title('PCR Drop Between Train and Validation Sets (AlphaFin)', fontsize=21)
plt.legend(fontsize=21)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.savefig('alphafin_consistency.png', dpi=700)  # Save with high resolution (300 DPI)
plt.show()