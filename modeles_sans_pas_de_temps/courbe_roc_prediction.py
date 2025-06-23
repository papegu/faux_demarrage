import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, roc_curve
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

# --- Fonctions statistiques ---
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

# --- Chargement et préparation des données ---
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

# --- Validation croisée k-fold ---
n_splits = 5
epochs = 50
batch_size = 32

history_per_fold = []
metrics_per_fold = []

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_no = 1
for train_idx, val_idx in skf.split(X_scaled, y):
    print(f"\n--- Fold {fold_no} ---")

    X_train_fold = X_scaled[train_idx]
    y_train_fold = y[train_idx]
    X_val_fold = X_scaled[val_idx]
    y_val_fold = y[val_idx]

    X_train_3d = X_train_fold.reshape((X_train_fold.shape[0], timesteps, features))
    X_val_3d = X_val_fold.reshape((X_val_fold.shape[0], timesteps, features))

    # Construction modèle
    input_layer = tf.keras.Input(shape=(timesteps, features))
    lstm_out = tf.keras.layers.LSTM(48)(input_layer)
    gru_out = tf.keras.layers.GRU(64)(input_layer)
    x_trans = tf.keras.layers.Dense(32)(input_layer)
    attn_out = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=16)(x_trans, x_trans)
    trans_out = tf.keras.layers.GlobalAveragePooling1D()(attn_out)
    concat = tf.keras.layers.concatenate([lstm_out, gru_out, trans_out])
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=4.99e-3), loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train_3d, y_train_fold,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_3d, y_val_fold),
        verbose=2
    )
    history_per_fold.append(history)

    acc, auc, prec, rec, f1, y_val_pred_prob = compute_metrics(model, X_val_3d, y_val_fold)
    print(f"Fold {fold_no} — Accuracy: {acc*100:.2f}%, AUC: {auc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

    metrics_per_fold.append((acc, auc, prec, rec, f1))

    # Tracé courbe ROC pour ce fold
    fpr, tpr, thresholds = roc_curve(y_val_fold, y_val_pred_prob)
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'ROC Fold {fold_no} (AUC={auc:.3f})', color='darkcyan')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Courbe ROC - Fold {fold_no}')
    plt.legend()
    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/roc_fold_{fold_no}.png", dpi=300)
    plt.close()

    # Sauvegarde données ROC dans CSV
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})
    roc_df.to_csv(f"figures/roc_data_fold_{fold_no}.csv", index=False)

    # Tracé courbe probabilités de faux démarrage triées
    idx_sorted = np.argsort(y_val_pred_prob)[::-1]
    sorted_pred = y_val_pred_prob[idx_sorted]
    sorted_true = y_val_fold[idx_sorted]

    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(sorted_pred)+1), sorted_pred, label='Probabilité de faux démarrage', color='darkcyan')
    colors = ['red' if val == 1 else 'grey' for val in sorted_true]
    plt.scatter(range(1, len(sorted_pred)+1), [0]*len(sorted_pred), color=colors, marker='|', label='Vérité terrain')
    plt.xlabel('Échantillons triés par probabilité décroissante')
    plt.ylabel('Probabilité de faux démarrage')
    plt.title(f'Courbe des probabilités de faux démarrage - Fold {fold_no}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"figures/courbe_faux_demarrage_fold_{fold_no}.png", dpi=300)
    plt.close()

    # Sauvegarde données probabilités triées
    pred_df = pd.DataFrame({'probabilité': sorted_pred, 'vrai_label': sorted_true})
    pred_df.to_csv(f"figures/probas_faux_demarrage_fold_{fold_no}.csv", index=False)

    fold_no += 1

# Moyenne et écart-type des métriques
metrics_array = np.array(metrics_per_fold)
mean_metrics = np.mean(metrics_array, axis=0)
std_metrics = np.std(metrics_array, axis=0)
metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1']

print("\n--- Résultats moyens et écart-type ---")
for i, name in enumerate(metric_names):
    if name == 'Accuracy':
        print(f"{name}: {mean_metrics[i]*100:.2f}% ± {std_metrics[i]*100:.2f}%")
    else:
        print(f"{name}: {mean_metrics[i]:.3f} ± {std_metrics[i]:.3f}")

# Tracé courbes accuracy par fold
plt.figure(figsize=(10, 6))
for i, history in enumerate(history_per_fold):
    plt.plot(history.history['accuracy'], label=f'Train Fold {i+1}')
    plt.plot(history.history['val_accuracy'], linestyle='--', label=f'Val Fold {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy per Fold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("figures/accuracy_cv_folds.png", dpi=300)
plt.show()
print("Graph accuracy CV saved in figures/accuracy_cv_folds.png")
# --- Affichage des prédictions vs vraie valeur (dernier fold) ---

# Pour le dernier fold traité (fold_no-1)
last_fold_idx = fold_no - 2  # comme fold_no a été incrémenté après la boucle

# On récupère les données du dernier fold
X_val_fold = X_scaled[val_idx]
y_val_fold = y[val_idx]
X_val_3d = X_val_fold.reshape((X_val_fold.shape[0], timesteps, features))

# Prédictions sur dernier fold
y_val_pred_prob = model.predict(X_val_3d, verbose=0).flatten()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_val_pred_prob)), y_val_pred_prob, c=y_val_fold, cmap='coolwarm', label='Probabilités')
plt.colorbar(label='Vraie valeur de faux démarrage')
plt.xlabel('Échantillons (index dans validation set)')
plt.ylabel('Probabilité prédite de faux démarrage')
plt.title('Probabilités prédites en fonction des vraies valeurs (dernier fold)')
plt.tight_layout()
plt.savefig("figures/predictions_vs_true_last_fold.png", dpi=300)
plt.show()
print("Figure saved as figures/predictions_vs_true_last_fold.png")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

plt.figure(figsize=(10, 6))

for fold, (y_val, y_pred_prob) in enumerate(roc_data_per_fold, start=1):
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    auc = roc_auc_score(y_val, y_pred_prob)
    plt.plot(fpr, tpr, label=f'Fold {fold} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC de l’étude en validation croisée')
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_all_folds.png", dpi=300)
