"""
Pipeline d'augmentation par GAN pour la detection des faux demarrages
- Correction de la fuite de donnees (leakage) :
    1. Split train/test AVANT tout calcul (scaler, tests statistiques, GAN)
    2. Scaler et tests statistiques fit UNIQUEMENT sur le train
    3. GAN entraine UNIQUEMENT sur le train
    4. Donnees synthetiques relabellisees avec les VRAIES regles
       agroclimatiques (pas un label constant)
    5. Le test set reste 100% reel, jamais enrichi de synthetique
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau
from tqdm import tqdm

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# ------------------------------------------------------------------------------
# Tests statistiques (Pettitt, Lombard, Kendall)
def pettitt_test(x):
    n = len(x)
    k_vals = []
    for k in range(1, n):
        s = 0
        for i in range(k):
            for j in range(k, n):
                s += np.sign(x[j] - x[i])
        k_vals.append(s)
    k_max = max(k_vals, key=abs)
    p = 2 * np.exp((-6 * k_max ** 2) / (n ** 3 - n))
    return p

def lombard_test(x):
    """
    NOTE : implementation a verifier / a remplacer par une reference
    academique standard avant publication (voir remarque precedente).
    Conservee ici a l'identique pour ne pas modifier ta methodologie
    sans validation prealable.
    """
    n = len(x)
    k_vals = np.arange(1, n)
    S = np.sum(k_vals * np.sign(np.array(x[1:]) - x[0]))
    p = 2 * np.exp(-S**2 / n)
    return p

def compute_statistical_tests(df_subset, variables):
    """Calcule Kendall / Pettitt / Lombard sur un sous-ensemble donne
    (train uniquement) et retourne un dict de valeurs a reappliquer
    de maniere identique (constante) sur train et sur les donnees
    synthetiques derivees du train."""
    stats = {}
    for var in variables:
        stats[f'p_Kendall_{var}'] = kendalltau(np.arange(len(df_subset)), df_subset[var])[1]
        stats[f'p_Pettitt_{var}'] = pettitt_test(df_subset[var].tolist())
        stats[f'p_Lombard_{var}'] = lombard_test(df_subset[var].tolist())
    return stats

# ------------------------------------------------------------------------------
# Regle de labellisation physique (identique a celle decrite dans le papier)
def compute_faux_demarage(row):
    """
    faux_demarage = 1 si :
      - cumul precipitation > 600 mm ET (sos_doy < 145 OU sos_doy > 180)
      - OU SPEI_1_July < -1  (stress hydrique significatif en juillet)
    sinon 0.
    Adapter les noms de colonnes exacts a ton dataset si besoin.
    """
    precip_cond = (row['PRECTOTCORR_SUM'] > 600) and (
        (row['sos_doy'] < 145) or (row['sos_doy'] > 180)
    )
    stress_cond = row['SPEI_1_July'] < -1
    return 1 if (precip_cond or stress_cond) else 0

# ------------------------------------------------------------------------------
# 1) Chargement des donnees
df = pd.read_excel("donnees_avec_faux_demarage.xlsx")

if 'faux_demarage' not in df.columns:
    raise ValueError("Colonne 'faux_demarage' absente")

required_cols = ['PRECTOTCORR_SUM', 'T2M', 'sos_doy', 'SPEI_1_July']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes necessaires manquantes pour la relabellisation : {missing}")

# ------------------------------------------------------------------------------
# 2) Split AVANT tout traitement (etape cle pour eviter le leakage)
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=RANDOM_STATE, stratify=df['faux_demarage']
)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

print(f"Train reel : {len(train_df)} annees | Test reel : {len(test_df)} annees")

# ------------------------------------------------------------------------------
# 3) Tests statistiques calcules UNIQUEMENT sur le train
variables_test = ['PRECTOTCORR_SUM', 'T2M']
train_stats = compute_statistical_tests(train_df, variables_test)

for col, val in train_stats.items():
    train_df[col] = val
    test_df[col] = val   # meme valeur figee (calculee sur train), appliquee au test

# ------------------------------------------------------------------------------
# 4) Scaler fit UNIQUEMENT sur le train
feature_cols = [c for c in train_df.columns if c != 'faux_demarage']

X_train_raw = train_df[feature_cols].values
y_train = train_df['faux_demarage'].values

X_test_raw = test_df[feature_cols].values
y_test = test_df['faux_demarage'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)   # transform seulement, jamais fit

# ------------------------------------------------------------------------------
# 5) GAN entraine UNIQUEMENT sur le train reel
latent_dim = 10
feature_dim = X_train_scaled.shape[1]
batch_size = min(16, len(X_train_scaled))   # adapte a un train de ~20 lignes
epochs_gan = 2000

def build_generator():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=latent_dim),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(feature_dim, activation='linear')
    ])

def build_discriminator():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(feature_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
discriminator.trainable = False

gan_input = tf.keras.Input(shape=(latent_dim,))
fake_features = generator(gan_input)
validity = discriminator(fake_features)
gan = tf.keras.Model(gan_input, validity)
gan.compile(optimizer='adam', loss='binary_crossentropy')

real_labels = np.ones((batch_size, 1))
fake_labels = np.zeros((batch_size, 1))

for epoch in tqdm(range(epochs_gan)):
    idx = np.random.randint(0, X_train_scaled.shape[0], batch_size)
    real_samples = X_train_scaled[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_samples = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    if epoch % 500 == 0:
        d_real = d_loss_real[0] if isinstance(d_loss_real, (list, np.ndarray)) else d_loss_real
        d_fake = d_loss_fake[0] if isinstance(d_loss_fake, (list, np.ndarray)) else d_loss_fake
        print(f"Epoch {epoch}/{epochs_gan} - D_real:{d_real:.4f} - D_fake:{d_fake:.4f} - G:{g_loss:.4f}")

# ------------------------------------------------------------------------------
# 6) Generation d'echantillons synthetiques (a partir du train uniquement)
n_synth = 24000
noise = np.random.normal(0, 1, (n_synth, latent_dim))
X_synth_scaled = generator.predict(noise, verbose=0)

X_synth_raw = scaler.inverse_transform(X_synth_scaled)
df_synth = pd.DataFrame(X_synth_raw, columns=feature_cols)

# ------------------------------------------------------------------------------
# 7) Relabellisation par les REGLES PHYSIQUES (jamais un label constant)
df_synth['faux_demarage'] = df_synth.apply(compute_faux_demarage, axis=1)

print("Repartition des classes - train reel     :", np.bincount(y_train.astype(int)))
print("Repartition des classes - synthetique     :", df_synth['faux_demarage'].value_counts().to_dict())

# ------------------------------------------------------------------------------
# 8) Validation de plausibilite (a inspecter avant d'utiliser les donnees)
print("\nComparaison des distributions (train reel vs synthetique) :")
for var in ['PRECTOTCORR_SUM', 'T2M', 'SPEI_1_July']:
    if var in df_synth.columns:
        print(f"  {var:20s} | reel  mean={train_df[var].mean():.2f} std={train_df[var].std():.2f}"
              f"  | synth mean={df_synth[var].mean():.2f} std={df_synth[var].std():.2f}")

# ------------------------------------------------------------------------------
# 9) Jeu d'entrainement final = train reel + synthetique correctement labellise
#    Le TEST reste 100% reel, jamais touche par le synthetique.
train_augmented_df = pd.concat(
    [train_df[feature_cols + ['faux_demarage']], df_synth[feature_cols + ['faux_demarage']]],
    ignore_index=True
)

# ------------------------------------------------------------------------------
# 10) Sauvegarde : train augmente + test reel dans des feuilles separees
with pd.ExcelWriter("data_augmentee_corrigee.xlsx") as writer:
    train_augmented_df.to_excel(writer, sheet_name="train_augmente", index=False)
    test_df[feature_cols + ['faux_demarage']].to_excel(writer, sheet_name="test_reel", index=False)

print("\nFichier genere : data_augmentee_corrigee.xlsx")
print(f"  - Feuille 'train_augmente' : {len(train_augmented_df)} lignes (reel + synthetique)")
print(f"  - Feuille 'test_reel'      : {len(test_df)} lignes (100% reel, jamais augmente)")
