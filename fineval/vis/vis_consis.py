import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("fineval_consistency.csv")
print(df.head())


# Define the bar width and positions
bar_width = 0.3
index = np.arange(len(df))

# Create the plot
plt.figure(figsize=(10, 6))
plt.barh(index, df['Drop_Train_Val'], height=bar_width, label='Drop', align='center',color="darkseagreen", alpha=0.7,linestyle='-')

# Set the y-axis ticks and labels
plt.yticks(index + bar_width / 2, df['Model'],fontsize=16)
plt.xticks(fontsize=16)
plt.xlim(-0.050, 0.15)
# Add labels and title
plt.title('Consistency Drop Between Train and Validation Sets (FinEval)', fontsize=16)
plt.legend()
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Show the plot
plt.tight_layout()
plt.savefig('./draws/fineval_consistency.png', dpi=700)  # Save with high resolution (300 DPI)
plt.show()