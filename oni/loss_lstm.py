import matplotlib.pyplot as plt

# Exemple de valeurs de loss (à remplacer par tes vraies données)
epochs = list(range(1, 35))  # Remplace par le nombre réel d'époques
train_loss = [0.2011, 0.0898, 0.0989, 0.0369, 0.0368, 0.0339, 0.0343, 0.0212, 0.0282, 0.0286, 
              0.0186, 0.0207, 0.0159, 0.0128, 0.0118, 0.0120, 0.0107, 0.0088, 0.0081, 0.0070,
              0.0067, 0.0072, 0.0052, 0.0072, 0.0093, 0.0047, 0.0064, 0.0067, 0.0102, 0.0058,
              0.0056, 0.0062, 0.0049, 0.0043]  # Remplace par tes vraies valeurs
val_loss = [0.2686, 0.0776, 0.1084, 0.1300, 0.0994, 0.0668, 0.0695, 0.0994, 0.1130, 0.0689, 
            0.0501, 0.0526, 0.0517, 0.0662, 0.0670, 0.0572, 0.0502, 0.0466, 0.0345, 0.0307, 
            0.0348, 0.0338, 0.0239, 0.0181, 0.0245, 0.0475, 0.0417, 0.0209, 0.0251, 0.0363,
            0.0411, 0.0326, 0.0284, 0.0359]  # Remplace par tes vraies valeurs

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='s')
plt.xlabel("Époques")
plt.ylabel("Loss")
plt.title("Évolution du Loss pour LSTM Bidirectionnel")
plt.legend()
plt.grid()
plt.show()
