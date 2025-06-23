import matplotlib.pyplot as plt
import numpy as np

# Values for each timestep
timesteps = [1, 2, 3, 4, 5]
accuracy = [0.71, 0.80, 0.86, 0.81, 0.85]
auc = [0.70, 0.93, 0.77, 0.85, 0.73]
f1 = [0.43, 0.73, 0.53, 0.63, 0.67]

# Bar width
bar_width = 0.25
x = np.arange(len(timesteps))

# Plot all 3 in the same figure
fig, ax = plt.subplots(figsize=(10, 6))

bars_accuracy = ax.bar(x - bar_width, accuracy, color='skyblue', edgecolor='white', label='Accuracy')
bars_auc = ax.bar(x, auc, color='lightgreen', edgecolor='white', label='AUC')
bars_f1 = ax.bar(x + bar_width, f1, color='lightcoral', edgecolor='white', label='F1-score')

# Highlight the best timestep (2nd, index 1) by adding black borders
for bar in [bars_accuracy[1], bars_auc[1], bars_f1[1]]:
    bar.set_edgecolor('black')
    bar.set_linewidth(2)

ax.set_xticks(x)
ax.set_xticklabels(timesteps)
ax.set_xlabel('Timestep')
ax.set_ylabel('Score')
ax.set_title('Model Performance Based on Timestep')
ax.legend()

fig.tight_layout()
plt.show()
