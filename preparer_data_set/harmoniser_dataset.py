import pandas as pd

# Lire le fichier Excel (remplacez 'votre_fichier.xlsx' par le nom réel)
df = pd.read_excel('base_dataset_drought.xlsx')

# Remplacer les valeurs vides par la moyenne de chaque colonne
df_filled = df.fillna(df.mean(numeric_only=True))

# Sauvegarder le dataframe modifié dans un nouveau fichier Excel
df_filled.to_excel('dataset_rempli.xlsx', index=False)
