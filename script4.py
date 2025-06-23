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
    """Pettitt's test for change point in a time series."""
    x = np.array(x)  # assure que x est un ndarray
    
    n = len(x)
    if n < 2:
        return np.nan  # Pas de p-value avec 0 ou 1 élément
    
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
    """Lombard's test for trend in a time series."""
    n = len(x)
    if n < 2:
        return np.nan
    k_vals = np.arange(1, n)
    S = np.sum(k_vals * np.sign(np.array(x[1:]) - x[0]))

    p = 2 * np.exp(-S**2 / n)
    return p

# -----------------------------
# Chargement et préparation des données
df = pd.read_excel("donnees_avec_faux_demarage.xlsx")

# Vérification colonne cible
if 'faux_demarage' not in df.columns:
    raise ValueError("Colonne 'faux_demarage' absente")

# Vérification de la présence des deux classes dans la cible
y = df['faux_demarage'].values
unique_classes = np.unique(y)
print(f"Classes présentes dans 'faux_demarage' : {unique_classes}")
if len(unique_classes) < 2:
    raise ValueError("La colonne 'faux_demarage' doit contenir au moins deux classes différentes (0 et 1).")

# -----------------------------
# Calcul p-values tests statistiques par variable sélectionnée
# Exemple sur 2 variables principales, adapte selon ton df
variables_test = ['PRECTOTCORR_SUM', 'T2M']

# Correction : nom de variable inutile dans la boucle
for var in variables_test:
    df[f'p_Kendall_{var}'] = kendalltau(np.arange(len(df)), df[var])[1]
    df[f'p_Pettitt_{var}'] = pettitt_test(df[var].tolist()) 
    df[f'p_Lombard_{var}'] = lombard_test(df[var].tolist()) 


X = df.drop(columns=['faux_demarage']).values
y = df['faux_demarage'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split avec stratification sécurisée
def safe_train_test_split(X, y, **kwargs):
    unique, counts = np.unique(y, return_counts=True)
    if np.any(counts < 2):
        kwargs.pop('stratify', None)
    return train_test_split(X, y, **kwargs)

X_train, X_test, y_train, y_test = safe_train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape pour entrée LSTM/GRU/Transformer : timesteps = 1
timesteps = 1
features = X_train.shape[1]
X_train_3d = X_train.reshape((X_train.shape[0], timesteps, features))
X_test_3d = X_test.reshape((X_test.shape[0], timesteps, features))

# -----------------------------
# Construction du modèle hybride LSTM+GRU+Transformer
input_layer = tf.keras.Input(shape=(timesteps, features))
lstm_out = tf.keras.layers.LSTM(32)(input_layer)
gru_out = tf.keras.layers.GRU(32)(input_layer)
x_trans = tf.keras.layers.Dense(32)(input_layer)
attn_out = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=8)(x_trans, x_trans)
trans_out = tf.keras.layers.GlobalAveragePooling1D()(attn_out)
concat = tf.keras.layers.concatenate([lstm_out, gru_out, trans_out])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat)

model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement
model.fit(X_train_3d, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

# -----------------------------
# Évaluation du modèle
def compute_metrics(model, X, y_true):
    y_pred_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = np.mean(y_pred == y_true)
    auc = roc_auc_score(y_true, y_pred_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, auc, prec, rec, f1, y_pred_prob

# Correction : éviter shadowing de noms de variables (acc, auc, etc.)
acc_val, auc_val, prec_val, rec_val, f1_val, y_pred_prob = compute_metrics(model, X_test_3d, y_test)
print(f'Hybrid Model Accuracy: {acc_val*100:.2f}%, AUC: {auc_val:.3f}, Precision: {prec_val:.3f}, Recall: {rec_val:.3f}, F1: {f1_val:.3f}')

# -----------------------------
# SHAP values : wrapper pour entrée 2D -> reshape 3D
def hybrid_predict_2d(X_2d):
    n_samples = X_2d.shape[0]
    X_3d = X_2d.reshape((n_samples, timesteps, features))
    return model.predict(X_3d, verbose=0).flatten()

# Limiter la taille pour rapidité
X_train_2d = X_train_3d.reshape((X_train_3d.shape[0], -1))
X_test_2d = X_test_3d.reshape((X_test_3d.shape[0], -1))

explainer = shap.KernelExplainer(hybrid_predict_2d, X_train_2d[:100])
shap_values = explainer.shap_values(X_test_2d[:100], nsamples=100)

# -----------------------------
# Bootstrapping SHAP values
def bootstrap_shap(explainer, X_data, n_iter=30, sample_size=100):
    shap_vals_collection = []
    for _ in tqdm(range(n_iter), desc="Bootstrapping SHAP"):
        idxs = np.random.choice(len(X_data), sample_size, replace=True)
        shap_vals = explainer.shap_values(X_data[idxs], nsamples=100)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        shap_vals_collection.append(shap_vals)
    return np.array(shap_vals_collection)

boot_shap_vals = bootstrap_shap(explainer, X_test_2d, n_iter=30, sample_size=100)

if isinstance(boot_shap_vals, list):
    boot_shap_vals = np.array(boot_shap_vals)

if boot_shap_vals.ndim == 3:
    # (n_bootstrap, n_samples, n_features) -> moyenne sur les samples
    mean_shap = np.mean(boot_shap_vals, axis=(0,1))
    std_shap = np.std(boot_shap_vals, axis=(0,1))
else:
    mean_shap = np.mean(boot_shap_vals, axis=0)
    std_shap = np.std(boot_shap_vals, axis=0)

print("Mean SHAP values shape:", mean_shap.shape)
print("Std SHAP values shape :", std_shap.shape)

# ---------------------------------------------------------------------------
# Summary Plot SHAP
# ---------------------------------------------------------------------------
os.makedirs("figures", exist_ok=True)
shap.summary_plot(shap_values, X_test_2d[:100], show=False)
plt.savefig("figures/shap_summary.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------------
# Bootstrapping SHAP Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()
ax.errorbar(range(mean_shap.shape[0]),
            mean_shap,
            yerr=std_shap,
            fmt='o',
            color='darkcyan',
            ecolor='lightgrey',
            capsize=5)
ax.set_xlabel('Features')
ax.set_ylabel('Mean SHAP ± Std')
ax.set_title('Bootstrapping SHAP Values')
fig.savefig("figures/shap_bootstrap.png", dpi=300)
plt.close()


def plot_faux_demarrage_curve(y_pred_prob, y_true=None):
    """
    Trace la courbe des probabilités de faux démarrage générées par le modèle.
    La courbe est triée par probabilité décroissante.
    Un scatter est ajouté en dessous afin d'indiquer le cas réel (1 ou 0).
    """
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
    fname = "figures/courbe_faux_demarrage.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"Graph sauvegardé dans {fname}")

# Correction : appel de plot_faux_demarrage_curve à la fin avec les bonnes variables
plot_faux_demarrage_curve(y_pred_prob, y_test)

# ---------------------------------------------------------------------------
# Validation croisée stratifiée et évaluation des performances des modèles
# ---------------------------------------------------------------------------
# Redéfinir X et y pour la validation croisée
X = df.drop(columns=['faux_demarage']).values
y = df['faux_demarage'].values

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Séquence pour LSTM/GRU/Transformer
timesteps = 1
features = X_scaled.shape[1]
X_seq = X_scaled.reshape((X_scaled.shape[0], timesteps, features))
y_seq = y  # Ajouté pour la boucle de validation croisée

# Configuration des modèles à tester
model_configs = {
    "LSTM": {"units": 32, "dropout": 0.2},
    "GRU": {"units": 32, "dropout": 0.2},
    "Transformer": {"num_heads": 4, "key_dim": 8, "dropout": 0.2}
}

results = []

# Validation croisée stratifiée
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Fonction de construction du modèle
def build_model(config):
    input_layer = tf.keras.Input(shape=(timesteps, features))
    if "units" in config:
        x = tf.keras.layers.LSTM(config["units"], dropout=config["dropout"])(input_layer)
    else:
        x = tf.keras.layers.GRU(config["units"], dropout=config["dropout"])(input_layer)
    if "num_heads" in config:
        x_trans = tf.keras.layers.Dense(32)(input_layer)
        attn_out = tf.keras.layers.MultiHeadAttention(num_heads=config["num_heads"], key_dim=config["key_dim"])(x_trans, x_trans)
        trans_out = tf.keras.layers.GlobalAveragePooling1D()(attn_out)
        x = tf.keras.layers.concatenate([x, trans_out])
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Boucle sur les configurations de modèles
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
        # Correction : gérer le cas où une seule classe est présente dans y_val
        try:
            auc = roc_auc_score(y_val, y_pred)
        except Exception:
            auc = float('nan')
        try:
            f1 = f1_score(y_val, y_pred_labels, zero_division=0)
        except Exception:
            f1 = float('nan')

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
# ---------------------------------------------------------------------------
# Fin du script

# Après le bootstrapping SHAP, pour le summary_plot :
# mean_shap.shape = (n_samples, n_features) ou (n_bootstrap, n_samples, n_features)
# X_val_2d doit avoir le même nombre de lignes que mean_shap

# Si mean_shap est (n_samples, n_features), alors n_samples = mean_shap.shape[0]
# Si mean_shap est (n_features,), reshape pour avoir (1, n_features)
# Correction du summary_plot SHAP pour éviter l'IndexError
# mean_shap doit avoir le même nombre de lignes que X_val_2d[:n_samples]
if mean_shap.ndim == 1:
    mean_shap = mean_shap.reshape(1, -1)

# Correction : feature_names doit être défini avant le summary_plot
feature_names = df.drop(columns=['faux_demarage']).columns.tolist()

# Vérification de la cohérence des dimensions pour summary_plot
n_samples = min(mean_shap.shape[0], X_test_2d.shape[0])
shap.summary_plot(mean_shap[:n_samples], X_test_2d[:n_samples], feature_names=feature_names, show=False)
plt.title("Summary Plot of Mean SHAP Values")
plt.savefig("figures/shap_summary.png", dpi=300)
plt.close()
