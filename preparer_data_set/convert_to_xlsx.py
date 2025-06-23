import pandas as pd
import os

# 🔁 Remplace le nom du fichier ici si besoin
input_file = 'senegal-and-mauritania-crops-vegetation-and-standardized-precipitation-index.csv'  # ou 'anomalies_precipitations.txt'

# 📦 Lire le fichier selon son extension
if input_file.endswith('.csv'):
    df = pd.read_csv(input_file)
elif input_file.endswith('.txt'):
    df = pd.read_csv(input_file, delim_whitespace=True, header=None)
    df.columns = ['year', 'value']  # Tu peux adapter les noms des colonnes

# 🔄 Déterminer le nom de sortie
output_file = os.path.splitext(input_file)[0] + '.xlsx'

# 💾 Exporter au format Excel
df.to_excel(output_file, index=False)

print(f"✅ Conversion terminée : {output_file}")
