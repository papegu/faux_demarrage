import pandas as pd

# Lecture du fichier ONI (structure SEAS, year, TOTAL, ANOM)
oni_df = pd.read_csv("oni.txt", sep=r"\s+", engine="python")

# Vérifie les colonnes
print("Colonnes du fichier :", oni_df.columns.tolist())

# Filtrage des années utiles
oni_df = oni_df[oni_df["year"].between(1981, 2020)]

# Calcul de la moyenne annuelle des anomalies (ONI)
oni_annual = oni_df.groupby("year")["ANOM"].mean().reset_index()
oni_annual.rename(columns={"ANOM": "oni"}, inplace=True)

# Sauvegarde en CSV et XLSX
oni_annual.to_csv("oni_annual_1981_2020.csv", index=False)
oni_annual.to_excel("oni_annual_1981_2020.xlsx", index=False)

print("Conversion réussie : 'oni_annual_1981_2020.csv' et '.xlsx' enregistrés.")
