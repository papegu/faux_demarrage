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
# 1️⃣ Chargement des données
df = pd.read_excel("donnees_avec_faux_demarage.xlsx")

if 'faux_demarage' not in df.columns:
    raise ValueError("Colonne 'faux_demarage' absente")

# ------------------------------------------------------------------------------
# 2️⃣ Tests statistiques sur le jeu d'origine (comme exemple)
variables_test = ['PRECTOTCORR_SUM', 'T2M']

for var in variables_test:
    df[f'p_Kendall_{var}'] = kendalltau(np.arange(len(df)), df[var])[1]
    df[f'p_Pettitt_{var}'] = pettitt_test(df[var].tolist()) 
    df[f'p_Lombard_{var}'] = lombard_test(df[var].tolist()) 


# ------------------------------------------------------------------------------
# 3️⃣ Préparation des données
X = df.copy()
if 'faux_demarage' in X.columns:
    X = X.drop(['faux_demarage'], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

input_dim = X_scaled.shape[1]
latent_dim = 32  # Dimension du bruit d'entrée
batch_size = 16
epochs = 5000

# ------------------------------------------------------------------------------
# 4️⃣ Generateur
def build_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, input_dim=latent_dim),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(512),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(input_dim, activation='linear')
    ])
    return model

# ------------------------------------------------------------------------------
# 5️⃣ Discriminateur
def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, input_dim=input_dim),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(256),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(128),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# ------------------------------------------------------------------------------
# 6️⃣ Compile
generator = build_generator()
discriminator = build_discriminator()
opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.5)

discriminator.compile(optimizer=opt, loss='binary_crossentropy')

z = tf.keras.Input(shape=(latent_dim,))
generated = generator(z)

discriminator.trainable = False
realistic = discriminator(generated)

gan = tf.keras.Model(z, realistic)
gan.compile(optimizer=opt, loss='binary_crossentropy')

# ------------------------------------------------------------------------------
# 7️⃣ Entraînement
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in tqdm(range(epochs)):
    idx = np.random.randint(0, X_scaled.shape[0], batch_size)
    real_samples = X_scaled[idx]

    z = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_samples = generator.predict(z)

    d_loss_real = discriminator.train_on_batch(real_samples, real)
    d_loss_fake = discriminator.train_on_batch(fake_samples, fake)

    z = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(z, real)

    if epoch % 500 == 0:
        print(f'Epoch {epoch} | d_real:{d_loss_real:.4f} | d_fake:{d_loss_fake:.4f} | g:{g_loss:.4f}')

# ------------------------------------------------------------------------------
# 8️⃣ Generation de 50 000 échantillons
z = np.random.normal(0, 1, (50000, latent_dim))
generated_data = generator.predict(z)

generated_df = scaler.inverse_transform(generated_data)
generated_df = pd.DataFrame(generated_df, columns=X.columns)

# ------------------------------------------------------------------------------
# 9️⃣ Tests statistiques sur le jeu généré
for var in variables_test:
    generated_df[f'p_Kendall_{var}'] = kendall_tau_val = kendalltau(np.arange(len(generated_df)), generated_df[var])[1]
    generated_df[f'p_Pettitt_{var}'] = pettitt_val = pettitt_test(generated_df[var].tolist()) 
    generated_df[f'p_Lombard_{var}'] = lombard_val = lombard_test(generated_df[var].tolist()) 


# ------------------------------------------------------------------------------
# 🔹 10️⃣ Sauvegarde
generated_df.to_excel("basedrought.xlsx", index=False)

print("Terminé. Fichier généré : basedrought.xlsx")
