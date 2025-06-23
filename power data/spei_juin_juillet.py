import pandas as pd

# === 1. Lecture du fichier SPEI
df = pd.read_csv("spei_sen.csv")

# === 2. Conversion de la colonne 'DATA' en format date
df['DATA'] = pd.to_datetime(df['DATA'], format='%b%Y')

# === 3. Extraire YEAR et MONTH
df['YEAR'] = df['DATA'].dt.year
df['MONTH'] = df['DATA'].dt.month

# === 4. Filtrer les années 2000 à 2016 et mois juin (6) et juillet (7)
df_filtered = df[(df['YEAR'].between(2000, 2016)) & (df['MONTH'].isin([6, 7]))]

# === 5. Garder les colonnes pertinentes
df_selected = df_filtered[['YEAR', 'MONTH', 'SPEI_1', 'SPEI_6']]

# === 6. Pivoter pour avoir une ligne par année
df_pivot = df_selected.pivot(index='YEAR', columns='MONTH', values=['SPEI_1', 'SPEI_6'])

# === 7. Renommer les colonnes
df_pivot.columns = ['SPEI_1_June', 'SPEI_1_July', 'SPEI_6_June', 'SPEI_6_July']
df_pivot.reset_index(inplace=True)  # Remettre YEAR en colonne

# === 8. Sauvegarde dans un fichier Excel
df_pivot.to_excel("spei_indices_2000_2016.xlsx", index=False)

print("✅ Fichier 'spei_indices_2000_2016.xlsx' enregistré avec succès.")
