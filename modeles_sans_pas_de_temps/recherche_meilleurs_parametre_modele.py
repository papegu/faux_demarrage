import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import kendalltau
from tqdm import tqdm
import shap
import matplotlib.pyplot as plt
import kerastuner as kt  # Ajout Keras Tuner

# -----------------------------
# Fonctions des tests statistiques (identiques)
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

# -----------------------------
# Chargement et préparation des données
df = pd.read_excel("donnees_avec_faux_demarage.xlsx")
if 'faux_demarage' not in df.columns:
    raise ValueError("Colonne 'faux_demarage' absente")

variables_test = ['PRECTOTCORR_SUM', 'T2M']
for var in variables_test:
    df[f'p_Kendall_{var}'] = kendalltau(np.arange(len(df)), df[var])[1]
    df[f'p_Pettitt_{var}'] = pettitt_test(df[var].tolist())
    df[f'p_Lombard_{var}'] = lombard_test(df[var].tolist())

X = df.drop(columns=['faux_demarage']).values
y = df['faux_demarage'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

timesteps = 1
features = X_scaled.shape[1]

def compute_metrics(model, X, y_true):
    y_pred_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = np.mean(y_pred == y_true)
    auc = roc_auc_score(y_true, y_pred_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, auc, prec, rec, f1, y_pred_prob

# -----------------------------
# Définition du modèle avec hyperparamètres variables pour Keras Tuner
def build_model(hp):
    input_layer = tf.keras.Input(shape=(timesteps, features))
    
    # Hyperparamètres tunés
    lstm_units = hp.Int('lstm_units', min_value=16, max_value=64, step=16)
    gru_units = hp.Int('gru_units', min_value=16, max_value=64, step=16)
    num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
    key_dim = hp.Choice('key_dim', values=[4, 8, 16])
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    
    lstm_out = tf.keras.layers.LSTM(lstm_units)(input_layer)
    gru_out = tf.keras.layers.GRU(gru_units)(input_layer)
    
    x_trans = tf.keras.layers.Dense(32)(input_layer)
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x_trans, x_trans)
    trans_out = tf.keras.layers.GlobalAveragePooling1D()(attn_out)
    
    concat = tf.keras.layers.concatenate([lstm_out, gru_out, trans_out])
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat)
    
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# -----------------------------
# Split train/validation pour tuner (par exemple 80/20)
X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
    X_scaled, y, stratify=y, test_size=0.2, random_state=42)

X_train_3d = X_train_tune.reshape((-1, timesteps, features))
X_val_3d = X_val_tune.reshape((-1, timesteps, features))

# -----------------------------
# Lancement de la recherche hyperparamétrique avec Keras Tuner
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    directory='kt_tuning',
    project_name='faux_demarage'
)

tuner.search(
    X_train_3d, y_train_tune,
    epochs=20,
    validation_data=(X_val_3d, y_val_tune),
    batch_size=32,
    verbose=2
)

best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Meilleurs hyperparamètres trouvés :")
print(f"LSTM units: {best_hp.get('lstm_units')}")
print(f"GRU units: {best_hp.get('gru_units')}")
print(f"Num heads: {best_hp.get('num_heads')}")
print(f"Key dim: {best_hp.get('key_dim')}")
print(f"Learning rate: {best_hp.get('learning_rate')}")

# -----------------------------
# Entraînement final du modèle avec les meilleurs hyperparamètres
model = build_model(best_hp)

history = model.fit(
    X_train_3d, y_train_tune,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_3d, y_val_tune),
    verbose=2
)

# Évaluation finale
acc, auc, prec, rec, f1, y_pred_prob_val = compute_metrics(model, X_val_3d, y_val_tune)
print(f"\nRésultats finaux sur validation :\nAccuracy: {acc*100:.2f}%, AUC: {auc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

# -----------------------------
# Le reste du script (plots, SHAP, courbes etc.) peut rester identique
# N'oublie pas de reshaper X_val_tune pour les SHAP etc si besoin
