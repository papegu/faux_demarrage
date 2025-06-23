import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Dossier contenant les fichiers Excel
folder_path = r'C:/Users/M.Gueye/Desktop/mes recherches en doctorat/projet drought/power data/powercsv'

# Liste tous les fichiers .xlsx dans le dossier
excel_files = glob.glob(os.path.join(folder_path, '*.xlsx'))

# Dictionnaire pour stocker les données annuelles par paramètre
data_per_param = {}

for file in excel_files:
    print(f"Lecture de {file}")
    df = pd.read_excel(file)
    
    # Remplacer les virgules par points dans les colonnes numériques au cas où
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '.')
    
    # Convertir les colonnes de mois en float (en supposant qu'elles sont toutes en majuscules 3 lettres)
    mois = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    for m in mois:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors='coerce')

    # Calculer la colonne ANN (somme des mois présents)
    df['ANN'] = df[mois].sum(axis=1)

    # Pour chaque PARAMETER, agréger la moyenne annuelle par YEAR
    grouped = df.groupby(['PARAMETER', 'YEAR'])['ANN'].mean().reset_index()

    # Organiser les données dans un format large (index YEAR, colonnes PARAMETER)
    pivot = grouped.pivot(index='YEAR', columns='PARAMETER', values='ANN')

    # Ajouter ou concaténer dans data_per_param
    if data_per_param:
        # Joindre les colonnes (paramètres) supplémentaires
        data_per_param = pd.concat([data_per_param, pivot], axis=1)
        # Enlever doublons éventuels de colonnes
        data_per_param = data_per_param.loc[:,~data_per_param.columns.duplicated()]
    else:
        data_per_param = pivot

# Supprimer les années avec trop de données manquantes
data_per_param = data_per_param.dropna(how='any')

# Calcul de la matrice de corrélation
matrice_corr = data_per_param.corr()

# Affichage heatmap
plt.figure(figsize=(10,8))
sns.heatmap(matrice_corr, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=0.5)
plt.title('Correlation matrix between indices based on annual trend')

plt.show()
