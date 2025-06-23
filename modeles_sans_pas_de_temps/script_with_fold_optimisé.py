import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import kendalltau
from tqdm import tqdm
import shap
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

# --- Validation croisée statique k-fold ---
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
    opt = tf.keras.optimizers.Adam(learning_rate=4.99e-3)
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train_3d, y_train_fold,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_3d, y_val_fold),
        verbose=2
    )
    history_per_fold.append(history)

    acc, auc, prec, rec, f1, _ = compute_metrics(model, X_val_3d, y_val_fold)
    print(f"Fold {fold_no} — Accuracy: {acc*100:.2f}%, AUC: {auc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")

    metrics_per_fold.append((acc, auc, prec, rec, f1))
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

# Tracé des courbes d'accuracy par fold
plt.figure(figsize=(10, 6))
for i, history in enumerate(history_per_fold):
    plt.plot(history.history['accuracy'], label=f'Train Fold {i+1}')
    plt.plot(history.history['val_accuracy'], linestyle='--', label=f'Val Fold {i+1}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy per Fold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/accuracy_cv_folds.png", dpi=300)
plt.show()
print("Graph accuracy CV saved in figures/accuracy_cv_folds.png")

# --- SHAP values sur le dernier modèle (dernier fold) ---

def hybrid_predict_2d(X_2d):
    n_samples = X_2d.shape[0]
    X_3d = X_2d.reshape((n_samples, timesteps, features))
    return model.predict(X_3d, verbose=0).flatten()

# Reshape pour SHAP (2D)
X_train_2d = X_train_3d.reshape((X_train_3d.shape[0], -1))
X_val_2d = X_val_3d.reshape((X_val_3d.shape[0], -1))

explainer = shap.KernelExplainer(hybrid_predict_2d, X_train_2d[:100])

shap_values = explainer.shap_values(X_val_2d[:100], nsamples=100)

def bootstrap_shap(explainer, X_data, n_iter=30, sample_size=100):
    shap_vals_collection = []
    for _ in tqdm(range(n_iter), desc="Bootstrapping SHAP"):
        idxs = np.random.choice(len(X_data), sample_size, replace=True)
        shap_vals = explainer.shap_values(X_data[idxs], nsamples=100)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals_collection.append(shap_vals)
    return np.stack(shap_vals_collection)

boot_shap_vals = bootstrap_shap(explainer, X_val_2d, n_iter=30, sample_size=100)

mean_shap = np.mean(boot_shap_vals, axis=0)
std_shap = np.std(boot_shap_vals, axis=0)

print("Mean SHAP values shape :", mean_shap.shape)
print("Std SHAP values shape :", std_shap.shape)
print("X_val_2d[:100] shape :", X_val_2d[:100].shape)

# Ajustement dimensions pour éviter l'erreur
min_samples = min(mean_shap.shape[0], X_val_2d[:100].shape[0])
min_features = min(mean_shap.shape[1], X_val_2d[:100].shape[1])

mean_shap_trimmed = mean_shap[:min_samples, :min_features]
X_val_2d_trimmed = X_val_2d[:min_samples, :min_features]

# Summary plot SHAP
shap.summary_plot(mean_shap_trimmed, X_val_2d_trimmed, show=False)
plt.title("Summary Plot of Mean SHAP Values")
plt.savefig("figures/shap_summary.png", dpi=300)
plt.close()

# Plot avec erreur (barres d’erreur)
fig, ax = plt.subplots()
ax.errorbar(range(mean_shap_trimmed.shape[1]),
            np.mean(mean_shap_trimmed, axis=0),
            yerr=np.std(mean_shap_trimmed, axis=0),
            fmt='o',
            color='darkcyan',
            ecolor='lightgrey',
            capsize=5)
ax.set_xlabel('Features')
ax.set_ylabel('Mean SHAP ± Std')
ax.set_title('Bootstrapped SHAP Values')

fig.savefig("figures/shap_bootstrap.png", dpi=300)
plt.close()

# --- Courbe de probabilité de faux démarrage ---
def plot_faux_demarrage_curve(y_pred_prob, y_true=None):
    idx_sorted = np.argsort(y_pred_prob)[::-1]
    sorted_pred = y_pred_prob[idx_sorted]
    if y_true is not None:
        sorted_true = y_true[idx_sorted]
    else:
        sorted_true = None
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_pred)+1), sorted_pred, label='Probabilité de faux démarrage', color='darkcyan')
    if sorted_true is not None:
        colors = ['red' if val == 1 else 'grey' for val in sorted_true]
        plt.scatter(range(1, len(sorted_pred)+1), [0]*len(sorted_pred), color=colors, marker='|', label='Réalité')
    plt.xlabel('Échantillons triés par probabilité décroissante')
    plt.ylabel('Probabilité de faux démarrage')
    plt.title('Courbe des probabilités de faux démarrage')
    if sorted_true is not None:
        plt.legend()
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fname = "figures/courbe_faux_demarrage.png"
    plt.savefig(fname, dpi=300)
    plt.show()
    print(f"Figure saved as {fname}")

# Affichage de la courbe sur la dernière validation
_, _, _, _, _, y_val_pred_prob = compute_metrics(model, X_val_3d, y_val_fold)
plot_faux_demarrage_curve(y_val_pred_prob, y_val_fold)
from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

def plot_confusion_and_calibration(y_true, y_prob, n_bins=10):
    import seaborn as sns

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
    bin_indices = np.digitize(y_prob, bins=np.linspace(0,1,n_bins+1), right=True)
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
    plt.bar(bin_stats_df['Mean Pred'], bin_stats_df['Count'] / max(bin_stats_df['Count']), 
            width=0.05, alpha=0.3, label='Distribution des prédictions (normalisée)')
    plt.xlabel('Probabilité prédite')
    plt.ylabel('Probabilité vraie')
    plt.title('Diagramme de Calibration')
    plt.legend()
    plt.tight_layout()
    plt.show()


# --- Utilisation de la fonction sur la dernière validation ---
plot_confusion_and_calibration(y_val_fold, y_val_pred_prob)

