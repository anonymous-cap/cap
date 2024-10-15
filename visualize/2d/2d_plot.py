import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return np.tanh((x + 0.01) / (y + 0.01))

def g(x,y):
    return np.tanh((x)/(y))



x = np.linspace(0, 1, 400)
y = np.linspace(0, 1, 400)
X, Y = np.meshgrid(x, y)
Z = g(X, Y)


plt.figure(figsize=(12, 6))
plt.contourf(X, Y, Z, 20, cmap='GnBu_r',alpha=1)
plt.colorbar()
plt.xlabel(' Performance Score', fontsize=14,fontweight='bold')
plt.ylabel('Consistency', fontsize=14, fontweight='bold')
plt.title('Performance Consistency Ratio', fontsize=17, fontweight='bold')


center_x, center_y = 0.5, 0.5
plt.plot(center_x, center_y, 'ko')
plt.text(center_x + 0.02, center_y + 0.02, 'Train Set', color='black', fontsize=18, fontweight='bold')


test_x1, test_y1 = 0.8, 0.15
#plt.plot(test_x1, test_y1, 'ko')
plt.text(0.74, 0.09, 'Test Set', color='black', fontsize=18,fontweight='bold')


test_x2, test_y2 = 0.22, 0.78
#plt.plot(test_x2, test_y2, 'ko')
plt.text(test_x2 - 0.1, test_y2 + 0.02, 'Test Set', color='black', fontsize=18,fontweight='bold')


plt.arrow(center_x, center_y, test_x1 - center_x, test_y1 - center_y, head_width=0.02, head_length=0.03, fc='black', ec='gray',linestyle='--')

plt.text(0.6, 0.3, '✘', color='red', fontsize=20, fontweight='bold')
plt.text(0.45, 0.25, 'Data Contamination', color='red', fontsize=17,fontweight='bold')


plt.arrow(center_x, center_y, test_x2 - center_x, test_y2 - center_y, head_width=0.02, head_length=0.03, fc='black', ec='gray',linestyle='--')

plt.text(0.4, 0.6, '✔', color='green', fontsize=27, fontweight='bold')
plt.text(0.47, 0.6, 'Fine-tuning', color='green', fontsize=17, fontweight='bold')


plt.show()
