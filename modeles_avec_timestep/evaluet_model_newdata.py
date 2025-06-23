import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from scipy.stats import kendalltau
import joblib

# --- Optionnel : fonctions statistiques à réutiliser ---
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
def plot_confusion_and_calibration(y_true, y_prob, n_bins=10):
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm,
                       index=['True Negative', 'True Positive'],
                       columns=['Predicted Negative', 'Predicted Positive']))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['True Negative', 'True Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('Ground Truth')
    plt.xlabel('Prediction')
    plt.tight_layout()
    plt.show()

    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins=bin_edges, right=True)
    bin_stats = []
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        count = np.sum(mask)
        mean_pred = np.mean(y_prob[mask]) if count > 0 else np.nan
        mean_true = np.mean(y_true[mask]) if count > 0 else np.nan
        bin_stats.append({'Bin': i, 'Count': count, 'Mean Predicted Prob': mean_pred, 'Mean True Label': mean_true})

    bin_stats_df = pd.DataFrame(bin_stats)
    print("\nCalibration bin statistics:")
    print(bin_stats_df)

    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.bar(bin_stats_df['Mean Predicted Prob'],
            bin_stats_df['Count'] / max(bin_stats_df['Count']),
            width=0.05, alpha=0.3, label='Normalized Prediction Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Diagram')
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- Load data --- 
# --- Chargement des nouvelles données ---
new_data_file = "nouveau_fichier_climatique.xlsx"
df = pd.read_excel(new_data_file)

# Vérifie les colonnes nécessaires
variables_test = ['PRECTOTCORR_SUM', 'T2M']
for var in variables_test:
    if var not in df.columns:
        raise ValueError(f"Colonne '{var}' absente du fichier")

# Ajout des features statistiques (doivent être les mêmes que lors de l'entraînement)
df[f'p_Kendall_PRECTOTCORR_SUM'] = kendalltau(np.arange(len(df)), df['PRECTOTCORR_SUM'])[1]
df[f'p_Kendall_T2M'] = kendalltau(np.arange(len(df)), df['T2M'])[1]
df[f'p_Pettitt_PRECTOTCORR_SUM'] = pettitt_test(df['PRECTOTCORR_SUM'].tolist())
df[f'p_Pettitt_T2M'] = pettitt_test(df['T2M'].tolist())
df[f'p_Lombard_PRECTOTCORR_SUM'] = lombard_test(df['PRECTOTCORR_SUM'].tolist())
df[f'p_Lombard_T2M'] = lombard_test(df['T2M'].tolist())

# --- Chargement du scaler sauvegardé (doit provenir de l'entraînement) ---
scaler = joblib.load("scaler.save")  # ← fichier sauvegardé à l'entraînement
X_new = scaler.transform(df.values)

# --- Reshape des données pour LSTM/GRU ---
timesteps = 2
features = X_new.shape[1]

def reshape_for_lstm(X, timesteps):
    n = len(X)
    if n % timesteps != 0:
        X = X[: (n // timesteps) * timesteps]
    return X.reshape((len(X) // timesteps, timesteps, features))

X_seq = reshape_for_lstm(X_new, timesteps)

# --- Prédiction avec les modèles sauvegardés ---
model_dir = "."  # Dossier courant ou dossier contenant les .h5
model_files = sorted([f for f in os.listdir(model_dir) if f.startswith("model_fold_") and f.endswith(".h5")])

all_preds = []

for model_file in model_files:
    print(f"Chargement du modèle {model_file}...")
    model = tf.keras.models.load_model(os.path.join(model_dir, model_file))
    y_pred_prob = model.predict(X_seq, verbose=0).flatten()

    # Aligne les prédictions à la forme d'origine
    full_pred = [np.nan] * (timesteps - 1) + y_pred_prob.tolist()
    df[f'Pred_{model_file.replace(".h5", "")}'] = full_pred[:len(df)]

    all_preds.append(y_pred_prob)

# --- Moyenne des prédictions (optionnel) ---
if all_preds:
    combined = np.stack([np.concatenate([[np.nan]*(timesteps-1), p]) for p in all_preds], axis=1)
    df['Mean_Prediction'] = np.nanmean(combined, axis=1)

# --- Sauvegarde des résultats ---
output_file = "resultats_predictions_nouvelles_donnees.xlsx"
df.to_excel(output_file, index=False)
print(f"✅ Résultats sauvegardés dans : {output_file}")

# --- Analyse globale sur toutes les validations ---
all_y_val = np.concatenate(all_y_val)
all_y_pred_prob = np.concatenate(all_y_pred_prob)

print("\n--- Résultats globaux sur toutes les validations ---")
plot_confusion_and_calibration(all_y_val, all_y_pred_prob, n_bins=10)   