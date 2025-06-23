import pandas as pd

# Lire le fichier Excel avec la colonne Date en format date
df = pd.read_excel('rainfallssr.xlsx', parse_dates=['Date'])
print(df.columns)

# Extraire l'année depuis la colonne Date
df['Year'] = df['Date'].dt.year

# Filtrer les années entre 2000 et 2017 inclus
df_filtered = df[(df['Year'] >= 2000) & (df['Year'] <= 2017)]

# Calculer la moyenne annuelle de SSRN
annual_mean = df_filtered.groupby('Year')['SSRN'].mean().reset_index()

# Exporter en fichier Excel
annual_mean.to_excel('rainfallssr1.xlsx', index=False)

print("✅ Fichier rainfallssr1.xlsx créé avec succès.")
