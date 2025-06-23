import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# ------------------------------------------------------------------------------
# 1️⃣ Chargement des données générées
df = pd.read_excel("donnees_avec_faux_demarage.xlsx")

# ------------------------------------------------------------------------------
# 2️⃣ Statistiques descriptives
print("Statistiques descriptives:")
print("----------------------------")
print(df.describe())  # Moy, écart-type, min, max, quartiles, etc.

# ------------------------------------------------------------------------------
# 3️⃣ Corrélations Pearson
print("\nMatrice de corrélations Pearson:")
print("----------------------------")
corr_matrix = df.corr()
print(corr_matrix)

# ------------------------------------------------------------------------------
# 4️⃣ Tests de normalité (Shapiro-Wilk) sur quelques variables
print("\nTest Shapiro-Wilk de normalité:")
print("----------------------------")
for column in ['PRECTOTCORR_SUM', 'T2M', 'RH2M']:  # vérifie quelques variables
    stat, p = shapiro(df[column].sample(25))  # 500 échantillons aléatoires
    print(f"{column}: W = {stat:.4f}, p-value = {p:.4f}")
    if p > 0.05:
        print("→ Distribution proche de la normalité.")
    else:
        print("→ Distribution non normale.")
    print()

# ------------------------------------------------------------------------------
# 5️⃣ Visualisation (optionnel)

# Histogrammes de quelques variables
for column in ['PRECTOTCORR_SUM', 'T2M', 'RH2M']:
    df[column].hist(bins=10)
    plt.title(f"Distribution de {column}")
    plt.xlabel(column)
    plt.ylabel("Fréquence")
    plt.show()

# ------------------------------------------------------------------------------
# 6️⃣ Heatmap de corrélations
sns.heatmap(corr_matrix, cmap='coolwarm', annot=False)
plt.title("Matrice de corrélations")
plt.show()
