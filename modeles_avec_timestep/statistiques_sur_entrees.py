import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_excel('donnees_avec_faux_demarage.xlsx')

# 1. Répartition de la cible binaire
print("Répartition de la cible 'faux_demarage':")
print(df['faux_demarage'].value_counts())
print("\nPourcentage:")
print(df['faux_demarage'].value_counts(normalize=True) * 100)

# Visualisation simple
sns.countplot(x='faux_demarage', data=df)
plt.title("Répartition des classes de la cible 'faux_demarage'")
plt.show()

# 2. Statistiques descriptives des variables d'entrée par classe cible
features = df.columns.drop('faux_demarage')

for feature in features:
    print(f"\nStatistiques de la variable '{feature}' par classe de 'faux_demarage':")
    print(df.groupby('faux_demarage')[feature].describe())

# Optionnel : boxplot pour visualiser les distributions selon la classe cible
for feature in features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='faux_demarage', y=feature, data=df)
    plt.title(f"Distribution de {feature} selon la classe 'faux_demarage'")
    plt.show()
