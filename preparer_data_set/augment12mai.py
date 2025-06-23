import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau
from tqdm import tqdm

# ------------------------------------------------------------------------------
# Tests statistiques
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
    n = len(x)
    k_vals = np.arange(1, n)
    S = np.sum(k_vals * np.sign(np.array(x[1:]) - x[0]))

    p = 2 * np.exp(-S**2 / n)
    return p

# ------------------------------------------------------------------------------
# Chargement des données
df = pd.read_excel("donnees_avec_faux_demarage.xlsx")

if 'faux_demarage' not in df.columns:
    raise ValueError("Colonne 'faux_demarage' absente")

# ------------------------------------------------------------------------------
# Tests statistiques
variables_test = ['PRECTOTCORR_SUM', 'T2M']

for var in variables_test:
    df[f'p_Kendall_{var}'] = kendalltau(np.arange(len(df)), df[var])[1]
    df[f'p_Pettitt_{var}'] = pettitt_test(df[var].tolist()) 
    df[f'p_Lombard_{var}'] = lombard_test(df[var].tolist())  

# ------------------------------------------------------------------------------
# Préparation des données
X = df.drop(columns=['faux_demarage']).values
y = df['faux_demarage'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------------------------------------------------------
# GAN
latent_dim = 10
feature_dim = X_scaled.shape[1]
batch_size = 32
epochs_gan = 2000

def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=latent_dim),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(feature_dim, activation='linear')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(feature_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

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

for epoch in range(epochs_gan):
    idx = np.random.randint(0, X_scaled.shape[0], batch_size)
    real_samples = X_scaled[idx]
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_samples = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
    d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, real_labels)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs_gan} - D_real_loss: {d_loss_real[0]:.4f} - D_fake_loss: {d_loss_fake[0]:.4f} - G_loss: {g_loss:.4f}")

# ------------------------------------------------------------------------------
# Generation de 24 000 échantillons synthétiques
n_synth = 24000
noise = np.random.normal(0, 1, (n_synth, latent_dim))
X_synth_scaled = generator.predict(noise)

# ------------------------------------------------------------------------------
# Etiquettes synthétiques
y_mode = 0 if np.mean(y) < 0.5 else 1
y_synth = np.full(n_synth, y_mode)

# ------------------------------------------------------------------------------
# Combine le jeu de données
X_combined = np.vstack([X_scaled, X_synth_scaled])
y_combined = np.concatenate([y, y_synth])

# ------------------------------------------------------------------------------
# Split et sauvegarde
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

df_synth = pd.DataFrame(scaler.inverse_transform(X_combined),
                         columns=df.columns.drop('faux_demarage'))
df_synth['faux_demarage'] = y_combined

df_synth.to_excel("data12mai.xlsx", index=False)

print("Fichier généré: data12mai.xlsx")
