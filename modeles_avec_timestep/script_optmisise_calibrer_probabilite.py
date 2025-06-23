import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.calibration import calibration_curve
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import seaborn as sns

# --- Tests statistiques (comme dans ton code d'origine) --- 
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

# --- Fonction pour afficher matrice de confusion + diagramme calibration --- 
def plot_confusion_and_calibration(y_true, y_prob, n_bins=10):
    # Binarisation des prédictions (seuil 0.5)
    y_pred = (y_prob >= 0.5).astype(int)

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nMatrice de Confusion:")
    print(pd.DataFrame(cm, 
                       index=['Vrai Negatif', 'Vrai Positif'], 
                       columns=['Prédit Negatif', 'Prédit Positif']))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Prédit Negatif', 'Prédit Positif'],
                yticklabels=['Vrai Negatif', 'Vrai Positif'])
    plt.title('Matrice de Confusion')
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.show()

    # Courbe de calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')

    # Calcul des effectifs et moyenne par bin
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins=bin_edges, right=True)
    bin_stats = []
    for i in range(1, n_bins+1):
        mask = bin_indices == i
        count = np.sum(mask)
        mean_pred = np.mean(y_prob[mask]) if count > 0 else np.nan
        mean_true = np.mean(y_true[mask]) if count > 0 else np.nan
        bin_stats.append({'Bin': i, 'Count': count, 'Mean Pred': mean_pred, 'Mean True': mean_true})

    bin_stats_df = pd.DataFrame(bin_stats)

    print("\nStatistiques par bin de calibration:")
    print(bin_stats_df)

    plt.figure(figsize=(8,6))
    plt.plot(prob_pred, prob_true, marker='o', label='Courbe de calibration')
    plt.plot([0,1], [0,1], linestyle='--', label='Parfaite calibration')
    # Affiche la distribution normalisée des prédictions par bin
    plt.bar(bin_stats_df['Mean Pred'], bin_stats_df['Count'] / max(bin_stats_df['Count']), 
            width=0.05, alpha=0.3, label='Distribution des prédictions (normalisée)')
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Probabilité vraie')
    plt.title('Diagramme de Calibration')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Chargement des données --- 
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

# --- Changement du nombre de timesteps --- 
timesteps = 2  # On passe de 1 à 2
features = X_scaled.shape[1]

def reshape_for_lstm(X, timesteps):
    n = len(X)
    if n % timesteps != 0:
        n_new = (n // timesteps) * timesteps
        X = X[:n_new]
    return X.reshape((len(X) // timesteps, timesteps, features))

def reshape_labels(y, timesteps):
    n = len(y)
    if n % timesteps != 0:
        n_new = (n // timesteps) * timesteps
        y = y[:n_new]
    return y[timesteps-1::timesteps]

X_lstm = reshape_for_lstm(X_scaled, timesteps)
y_lstm = reshape_labels(y, timesteps)

# --- Validation croisée --- 
n_splits = 5
epochs = 50
batch_size = 32

def compute_metrics(model, X, y_true):
    y_pred_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = np.mean(y_pred == y_true)
    auc = roc_auc_score(y_true, y_pred_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, auc, prec, rec, f1, y_pred_prob

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

metrics_per_fold = []
all_y_val = []
all_y_pred_prob = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_lstm, y_lstm), 1):
    X_train, X_val = X_lstm[train_idx], X_lstm[val_idx]
    y_train, y_val = y_lstm[train_idx], y_lstm[val_idx]

    # --- Construction du modèle --- 
    input_layer = tf.keras.Input(shape=(timesteps, features))
    lstm_out = tf.keras.layers.LSTM(48)(input_layer)
    gru_out = tf.keras.layers.GRU(64)(input_layer)
    x_trans = tf.keras.layers.Dense(32)(input_layer)
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)(x_trans, x_trans)
    trans_out = tf.keras.layers.GlobalAveragePooling1D()(attn_out)
    concat = tf.keras.layers.concatenate([lstm_out, gru_out, trans_out])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)

    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # --- Entraînement --- 
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=2
    )

    # --- Evaluation --- 
    acc, auc, prec, rec, f1, y_pred_prob = compute_metrics(model, X_val, y_val)

    print(f"Fold {fold} — Accuracy:{acc*100:.2f}%, AUC:{auc:.3f}, Precision:{prec:.3f}, Recall:{rec:.3f}, F1:{f1:.3f}")
    metrics_per_fold.append((acc, auc, prec, rec, f1))

    all_y_val.append(y_val)
    all_y_pred_prob.append(y_pred_prob)

print("\nMoyennes :", np.mean(metrics_per_fold, axis=0))
print("Écart-type :", np.std(metrics_per_fold, axis=0))

# --- Analyse globale sur toutes les validations --- 
all_y_val = np.concatenate(all_y_val)
all_y_pred_prob = np.concatenate(all_y_pred_prob)

print("\n--- Résultats globaux sur toutes les validations ---")
plot_confusion_and_calibration(all_y_val, all_y_pred_prob, n_bins=10)
