import pandas as pd

input_file = 'power_data_wind2m.csv'
output_file = 'power_data_wind2m.xlsx'

# Lire le fichier et ignorer les lignes d'en-tête personnalisées
with open(input_file, 'r') as f:
    lines = f.readlines()

# Trouver la fin de l'en-tête personnalisée
end_header_idx = 0
for i, line in enumerate(lines):
    if line.strip() == '-END HEADER-':
        end_header_idx = i
        break

# Lire les données CSV à partir de la ligne juste après '-END HEADER-'
data_lines = lines[end_header_idx+1:]

# Convertir la liste de lignes en DataFrame pandas
from io import StringIO
data_str = ''.join(data_lines)
df = pd.read_csv(StringIO(data_str))

# Exporter vers Excel
df.to_excel(output_file, index=False)

print(f"✅ Export terminé : {output_file}")
