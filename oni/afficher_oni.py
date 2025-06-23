import pandas as pd

# Charger en précisant le séparateur et en sautant les lignes vides
oni_df = pd.read_csv("oni.txt", delim_whitespace=True, skip_blank_lines=True)

print(oni_df.head())
print(oni_df.columns)
