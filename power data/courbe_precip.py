import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Dossier contenant les fichiers Excel
folder_path = r"C:\Users\M.Gueye\Desktop\mes recherches en doctorat\projet drought\power data\powercsv"

# Liste des fichiers Excel dans le dossier
files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx') or f.endswith('.xls')]

# Dictionnaire pour stocker les données annuelles par paramètre
annual_data = {}

for file in files:
    file_path = os.path.join(folder_path, file)
    print(f"Lecture du fichier : {file}")

    # Lire le fichier Excel
    df = pd.read_excel(file_path)

    # Nettoyage des colonnes : enlever espaces inutiles
    df.columns = df.columns.str.strip()

    # Vérifier si la colonne 'ANN' existe
    if 'ANN' in df.columns:
        # Extraire la colonne ANN par année et paramètre
        df_subset = df[['PARAMETER', 'YEAR', 'ANN']].copy()
    else:
        # Sinon calculer la somme des colonnes mensuelles (jan à dec)
        mois = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
        mois_existants = [m for m in mois if m in df.columns]
        if not mois_existants:
            print(f"Aucune colonne ANN ou mois dans {file}, fichier ignoré.")
            continue
        df['ANN'] = df[mois_existants].sum(axis=1)
        df_subset = df[['PARAMETER', 'YEAR', 'ANN']].copy()

    # Nettoyer la colonne YEAR : convertir en int (parfois années avec virgule ou espaces)
    df_subset['YEAR'] = df_subset['YEAR'].astype(str).str.replace(',', '').str.strip()
    df_subset = df_subset[df_subset['YEAR'].str.isnumeric()]  # garder que les années valides
    df_subset['YEAR'] = df_subset['YEAR'].astype(int)

    # Regrouper par PARAMETER et YEAR, moyenne pour éviter doublons d'années
    df_grouped = df_subset.groupby(['PARAMETER', 'YEAR'])['ANN'].mean().reset_index()

    # Stocker dans dict : liste par paramètre
    for param in df_grouped['PARAMETER'].unique():
        data_per_param = df_grouped[df_grouped['PARAMETER'] == param][['YEAR', 'ANN']].set_index('YEAR')
        if param in annual_data:
            # Fusionner sur l'index YEAR (union des années)
            annual_data[param] = annual_data[param].combine_first(data_per_param)
        else:
            annual_data[param] = data_per_param

# Fusionner tous les paramètres dans un seul DataFrame
df_all = pd.DataFrame()
for param, data in annual_data.items():
    data = data.rename(columns={'ANN': param})
    if df_all.empty:
        df_all = data
    else:
        df_all = df_all.join(data, how='outer')

# Supprimer les années où il n'y a pas assez de données (ex: années avec trop de NaN)
df_all = df_all.dropna(thresh=2)  # au moins 2 paramètres présents

print("Table finale des données annuelles :")
print(df_all.head())

# Calculer la matrice de corrélation (Pearson par défaut)
matrice_corr = df_all.corr()

# Afficher la heatmap
plt.figure(figsize=(10,8))
sns.heatmap(matrice_corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
plt.title("Matrice de corrélation entre indices annuels")
plt.show()
