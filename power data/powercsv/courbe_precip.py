import pandas as pd
import matplotlib.pyplot as plt

# Lire le fichier CSV (remplace le nom du fichier si nécessaire)
fichier = "power_data_precip.csv"

# Lire les données
df = pd.read_csv(fichier, sep="\t|,", engine="python")

# Vérifier que les colonnes sont bien lues
print(df.columns)

# Convertir les années en format entier
df["YEAR"] = df["YEAR"].astype(int)

# Agréger par année : somme des précipitations annuelles sur tous les points
precip_annuelles = df.groupby("YEAR")["ANN"].mean().reset_index()

# Tracer l'évolution
plt.figure(figsize=(10, 6))
plt.plot(precip_annuelles["YEAR"], precip_annuelles["ANN"], marker='o', linestyle='-', color='blue')
plt.title("Annual Precipitation Trend in Senegal")
plt.xlabel("Year")
plt.ylabel("Annual Precipitation (mm)")
plt.grid(True)
plt.tight_layout()
plt.show()
