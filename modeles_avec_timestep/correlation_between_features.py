import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le fichier Excel
df = pd.read_excel('donnees_avec_faux_demarage.xlsx')

# Calculer la matrice de corrélation
corr_matrix = df.corr()

# Afficher la matrice de corrélation
plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Matrice de corrélation des attributs")
plt.show()

# Enregistrer la matrice de corrélation dans un fichier Excel
corr_matrix.to_excel('matrice_correlation.xlsx')    