import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Lade die Kepler-Daten
X_train = np.load("X_train_kepler.npy")
y_train = np.load("y_train_kepler.npy")

# Normiere die Features für eine bessere t-SNE-Darstellung
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Reduziere die Dimension auf 2 mit t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Visualisiere die t-SNE-Reduktion
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[y_train == 0, 0], X_tsne[y_train == 0, 1], label="Nicht-Exoplaneten", alpha=0.5, color="blue")
plt.scatter(X_tsne[y_train == 1, 0], X_tsne[y_train == 1, 1], label="Exoplaneten", alpha=0.5, color="red")
plt.legend()
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE Visualisierung der Kepler-Daten")
plt.show()


# Falls die Features eine klare Trennung haben (erste Hälfte = Global View, zweite Hälfte = Local View)
num_features = X_train.shape[1]
split_index = num_features // 2  # Hälfte als Global View, Hälfte als Local View

global_view = X_train[:, :split_index]
local_view = X_train[:, split_index:]

def plot_tsne(X, y, title):
    if X.shape[1] == 0:
        print(f"⚠️ {title} konnte nicht erstellt werden, da die Daten keine gültigen Features haben.")
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], c='blue', label="Nicht-Exoplaneten", alpha=0.6, s=10)
    plt.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], c='red', label="Exoplaneten", alpha=0.6, s=10)
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()

# t-SNE für Global View
plot_tsne(global_view, y_train, "t-SNE Visualisierung der Global View")

# t-SNE für Local View
plot_tsne(local_view, y_train, "t-SNE Visualisierung der Local View")
