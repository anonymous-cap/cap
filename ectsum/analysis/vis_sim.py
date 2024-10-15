import matplotlib.pyplot as plt
import pandas as pd


data = pd.read_csv("ectsum_sim_clean_draw.csv")

models = data['model'].unique()
types = data['type'].unique()


colors = ['silver', 'darkseagreen', 'peachpuff', 'powderblue', 'moccasin', 'lightsteelblue', 'lavender']


if len(colors) < len(models):
    colors = colors * (len(models) // len(colors) + 1)


fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
for i, t in enumerate(types):
    for idx, model in enumerate(models):

        subset = data[(data['model'] == model) & (data['type'] == t)]


        axes[i].plot(subset['thresh'], subset['value'], marker='o', color=colors[idx], linestyle='-', markersize=8, linewidth=3, label=model)

    axes[i].set_title(f'{t.capitalize()} Set', fontsize=15)
    axes[i].set_xlabel('Threshold', fontsize=14)
    axes[i].set_ylabel('Consistency Score', fontsize=14)
    axes[i].grid(True)
    axes[i].legend(title='Model')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('./imgs/thre_consistency.png', dpi=300)
plt.show()