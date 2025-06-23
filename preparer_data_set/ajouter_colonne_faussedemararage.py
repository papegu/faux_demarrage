import pandas as pd

# Lire un fichier Excel correctement (sans delimiter)
df = pd.read_excel("dataset_rempli.xlsx")

# Conversion des colonnes : remplacer virgule par point puis convertir en float
for col in df.columns[1:]:  # on ignore l'année (colonne 0)
    df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
    df[col] = pd.to_numeric(df[col], errors='coerce')  # conversion en float

# Appliquer une règle simple pour détecter les faux démarrages
df["faux_demarage"] = df.apply(
    lambda row: 1 if (
        row["PRECTOTCORR_SUM"] > 600 and 
        (row["sos_doy"] < 145 or row["sos_doy"] > 180) or
        row["SPEI_1_July"] < -1
    ) else 0,
    axis=1
)

# Sauvegarder le nouveau fichier avec la colonne ajoutée
df.to_excel("donnees_avec_faux_demarage.xlsx", index=False)
print("✅ Colonne 'faux_demarage' ajoutée avec succès.")
