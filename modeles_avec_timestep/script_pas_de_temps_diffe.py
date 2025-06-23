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

# -----------------------------
# Fonctions des tests statistiques

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

# Variables à tester (à adapter)
variables_test = ['PRECTOTCORR_SUM', 'T2M']

for var in variables_test:
    df[f'p_Kendall_{var}'] = kendalltau(np.arange(len(df)), df[var])[1]
    df[f'p_Pettitt_{var}'] = pettitt_test(df[var].tolist())
    df[f'p_Lombard_{var}'] = lombard_test(df[var].tolist())

X = df.drop(columns=['faux_demarage']).values
y = df['faux_demarage'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 timesteps_vals = [1, 2, 3, 4, 5]

results = {ts: [] for ts in timesteps_vals}

for timesteps in timesteps_vals:
    print(f"{'='*20}\nÉvaluation avec TIMESTEP = {timesteps}\n{'='*20}")
    n_features = X_scaled.shape[1]

    # Prepare sequences
    X_seq = []
    y_seq = []

    for i in range(timesteps, len(X_scaled)):
        X_seq.append(X_scaled[i-timesteps : i, :]) 
        y_seq.append(y[i])

    X_seq = np.stack(X_seq)  # (samples, timesteps, features)
    y_seq = np.array(y_seq)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []

    fold = 1
    for train_idx, val_idx in skf.split(X_seq, y_seq):
        print(f"Fold {fold}")
        
        X_train, X_val = X_seq[train_idx], X_seq[val_idx]
        y_train, y_val = y_seq[train_idx], y_seq[val_idx]

        # Model definition
        input_layer = tf.keras.Input(shape=(timesteps, n_features))
        lstm_out = tf.keras.layers.LSTM(32)(input_layer)
        gru_out = tf.keras.layers.GRU(32)(input_layer)
        x_trans = tf.keras.layers.Dense(32)(input_layer)
        attn_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=8)(x_trans, x_trans)
        trans_out = tf.keras.layers.GlobalAveragePooling1D()(attn_out)

        concat = tf.keras.layers.concatenate([lstm_out, gru_out, trans_out])

        output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)

        model = tf.keras.Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            verbose=0)

        # Evaluation
        y_pred = model.predict(X_val, verbose=0).flatten()
        y_pred_labels = (y_pred >= 0.5).astype(int)

        acc = np.mean(y_pred_labels == y_val)
        auc = roc_auc_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred_labels, zero_division=0)

        fold_metrics.append((acc, auc, f1))
        fold += 1
    
    fold_metrics = np.array(fold_metrics)
    mean_vals = fold_metrics.mean(0)
    std_vals = fold_metrics.std(0)

    results[timesteps] = {
        'Accuracy_mean': mean_vals[0],
        'Accuracy_std': std_vals[0],
        'AUC_mean': mean_vals[1],
        'AUC_std': std_vals[1],
        'F1_mean': mean_vals[2],
        'F1_std': std_vals[2],
    }
    print(f"Timesteps {timesteps} - Accuracy: {mean_vals[0]:.4f}±{std_vals[0]:.4f}, AUC: {mean_vals[1]:.4f}±{std_vals[1]:.4f}, F1: {mean_vals[2]:.4f}±{std_vals[2]:.4f}") 
timesteps = 2
n_features = X_scaled.shape[1]

# Prepare sequences
X_seq = []
y_seq = []

for i in range(timesteps, len(X_scaled)):
    X_seq.append(X_scaled[i-timesteps : i, :]) 
    y_seq.append(y[i])

X_seq = np.stack(X_seq)  # (samples, timesteps, features)
y_seq = np.array(y_seq)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def build_model(architecture):
    """Construit le modèle en fonction de l'architecture demandée."""
    input_layer = tf.keras.Input(shape=(timesteps, n_features))
    components = []

    if "lstm" in architecture:
        components.append(tf.keras.layers.LSTM(32)(input_layer))
    if "gru" in architecture:
        components.append(tf.keras.layers.GRU(32)(input_layer))
    if "transformer" in architecture:
        x_trans = tf.keras.layers.Dense(32)(input_layer)
        attn_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=8)(x_trans, x_trans)
        components.append(tf.keras.layers.GlobalAveragePooling1D()(attn_out))
    
    concat = components[0] if len(components) == 1 else tf.keras.layers.concatenate(components)

    output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)

    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Liste des modèles à tester
model_configs = {
    "LSTM seul": ["lstm"],
    "LSTM + GRU": ["lstm", "gru"],
    "LSTM + Transformer": ["lstm", "transformer"],
    "GRU + Transformer": ["gru", "transformer"],
    "Complet (LSTM + GRU + Transformer)": ["lstm", "gru", "transformer"],
}

results = []

for name, config in model_configs.items():
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_seq, y_seq), start=1):
        X_train, X_val = X_seq[train_idx], X_seq[val_idx]
        y_train, y_val = y_seq[train_idx], y_seq[val_idx]

        model = build_model(config)
        history = model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            verbose=0)

        y_pred = model.predict(X_val, verbose=0).flatten()
        y_pred_labels = (y_pred >= 0.5).astype(int)

        acc = np.mean(y_pred_labels == y_val)
        auc = roc_auc_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred_labels, zero_division=0)

        fold_metrics.append([acc, auc, f1])

    fold_metrics = np.array(fold_metrics)
    mean_vals = fold_metrics.mean(0)
    std_vals = fold_metrics.std(0)

    results.append([name, 
                    f"{mean_vals[0]:.4f}±{std_vals[0]:.4f}",
                    f"{mean_vals[1]:.4f}±{std_vals[1]:.4f}",
                    f"{mean_vals[2]:.4f}±{std_vals[2]:.4f}"])

# Afficher le tableau des performances
df_result = pd.DataFrame(results, columns=['Modèle', 'Accuracy', 'AUC', 'F1-score'])

print("Tableau des performances:")
print(df_result)
