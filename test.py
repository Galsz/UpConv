"""
test.py — Executa todos os experimentos e salva os resultados em results/
===========================================================================
Autores: Geovane Araujo de Lima Silva (RA: 00111884)
         João Vitor Marinonio de Almeida

Uso:
    python test.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")  # sem janela gráfica, salva direto em arquivo
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from radial_profile import azimuthal_average, compute_power_spectrum_1d, rgb_to_gray
from spectral_utils import compute_mean_spectrum, plot_spectrum_comparison

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZE = 128
N_SAMPLES = 80
N_BINS = 60
np.random.seed(42)

# ---------------------------------------------------------------------------
# Geradores de imagens sintéticas
# ---------------------------------------------------------------------------

def natural_image(size=SIZE):
    noise = np.random.randn(size, size)
    blurred = cv2.GaussianBlur(noise, (21, 21), 6)
    return np.clip((blurred - blurred.min()) / (blurred.max() - blurred.min()), 0, 1)

def dcgan_image(size=SIZE):
    base = natural_image(size)
    idx = np.indices((size, size)).sum(axis=0) % 2
    return np.clip(base + 0.12 * idx, 0, 1)

def wgan_image(size=SIZE):
    base = natural_image(size)
    idx = np.indices((size, size)).sum(axis=0) % 4
    return np.clip(base + 0.07 * idx, 0, 1)

def extract_features(images, n_bins=N_BINS):
    return np.array([compute_power_spectrum_1d(img)[:n_bins] for img in images])

# ---------------------------------------------------------------------------
# Experimento 1 — Espectros FFT de padrões sintéticos
# ---------------------------------------------------------------------------

def exp1_espectros_sinteticos():
    print("  [1/5] Gerando espectros de padrões sintéticos...")

    x = np.linspace(0, 1, SIZE)
    xx, yy = np.meshgrid(x, x)

    gradiente = (xx + yy) / 2
    senoidal = 0.5 + 0.5 * np.sin(2 * np.pi * 8 * xx)
    ruido = np.random.rand(SIZE, SIZE)
    grade = np.zeros((SIZE, SIZE))
    grade[::4, :] = 1.0
    grade[:, ::4] = 1.0

    imagens = {
        "Gradiente Suave\n(Baixa Freq.)": gradiente,
        "Senoidal\n(Freq. Única)": senoidal,
        "Ruído Branco\n(Espectro Plano)": ruido,
        "Artefato Grade\n(Up-Conv)": grade,
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col, (nome, img) in enumerate(imagens.items()):
        axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(nome, fontsize=10)
        axes[0, col].axis("off")

        fshift = np.fft.fftshift(np.fft.fft2(img))
        mag = 20 * np.log(np.abs(fshift) + 1e-8)
        axes[1, col].imshow(mag, cmap="inferno")
        axes[1, col].set_title("Espectro FFT (dB)", fontsize=10)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("Domínio Espacial", fontsize=11)
    axes[1, 0].set_ylabel("Domínio Frequência", fontsize=11)
    fig.suptitle("Padrões Sintéticos e seus Espectros de Frequência",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "01_espectros_sinteticos.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     → Salvo: {path}")

# ---------------------------------------------------------------------------
# Experimento 2 — Perfis radiais
# ---------------------------------------------------------------------------

def exp2_perfis_radiais():
    print("  [2/5] Calculando perfis radiais azimutais...")

    x = np.linspace(0, 1, SIZE)
    xx, _ = np.meshgrid(x, x)
    gradiente = (xx + np.linspace(0, 1, SIZE).reshape(-1, 1)) / 2
    grade = np.zeros((SIZE, SIZE))
    grade[::4, :] = 1.0
    grade[:, ::4] = 1.0
    ruido = np.random.rand(SIZE, SIZE)
    senoidal = 0.5 + 0.5 * np.sin(2 * np.pi * 8 * xx)

    imagens = {
        "Gradiente": gradiente,
        "Senoidal": senoidal,
        "Ruído": ruido,
        "Grade (artefato)": grade,
    }
    cores = ["steelblue", "tomato", "mediumseagreen", "darkorange"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (nome, img), cor in zip(imagens.items(), cores):
        psd = compute_power_spectrum_1d(img)
        axes[0].plot(psd, label=nome, color=cor, linewidth=2)

    axes[0].set_xlabel("Frequência Radial (bin)", fontsize=12)
    axes[0].set_ylabel("Potência Normalizada", fontsize=12)
    axes[0].set_title("Perfis Espectrais Radiais 1D", fontsize=13)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    psd_grad = compute_power_spectrum_1d(gradiente)
    psd_grade = compute_power_spectrum_1d(grade)
    xr = np.arange(len(psd_grad))
    axes[1].plot(psd_grad, color="steelblue", label="Natural (Gradiente)", linewidth=2)
    axes[1].plot(psd_grade, color="tomato", label="GAN-like (Grade)", linewidth=2, linestyle="--")
    axes[1].fill_between(xr, psd_grad, psd_grade,
                         where=(psd_grade > psd_grad),
                         alpha=0.3, color="tomato", label="Excesso espectral")
    axes[1].set_xlabel("Frequência Radial (bin)", fontsize=12)
    axes[1].set_ylabel("Potência Normalizada", fontsize=12)
    axes[1].set_title("Distorção Espectral: Natural vs. GAN-like", fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Integração Azimutal — Perfis Radiais", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "02_perfis_radiais.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     → Salvo: {path}")

# ---------------------------------------------------------------------------
# Experimento 3 — Comparação espectral Real vs. GAN
# ---------------------------------------------------------------------------

def exp3_comparacao_espectros():
    print("  [3/5] Comparando espectros Real vs. GAN...")

    reais = [natural_image() for _ in range(N_SAMPLES)]
    gans  = [dcgan_image()   for _ in range(N_SAMPLES)]

    X_real = extract_features(reais)
    X_gan  = extract_features(gans)

    mean_r, std_r = X_real.mean(0), X_real.std(0)
    mean_g, std_g = X_gan.mean(0),  X_gan.std(0)

    x = np.arange(N_BINS)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, mean_r, color="steelblue", label="Imagens Reais", linewidth=2)
    ax.fill_between(x, mean_r - std_r, mean_r + std_r, alpha=0.3, color="steelblue")
    ax.plot(x, mean_g, color="tomato", label="Imagens GAN (DCGAN)", linewidth=2, linestyle="--")
    ax.fill_between(x, mean_g - std_g, mean_g + std_g, alpha=0.3, color="tomato")
    ax.set_xlabel("Frequência Radial (bin)", fontsize=12)
    ax.set_ylabel("Potência Normalizada", fontsize=12)
    ax.set_title("Espectro de Potência Azimutal: Real vs. GAN\n"
                 "(Reprodução da Figura 1 — Durall et al., CVPR 2020)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "03_comparacao_espectros.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     → Salvo: {path}")

    return X_real, X_gan

# ---------------------------------------------------------------------------
# Experimento 4 — Classificação e PCA
# ---------------------------------------------------------------------------

def exp4_classificacao(X_real, X_gan):
    print("  [4/5] Avaliando classificadores...")

    X = np.vstack([X_real, X_gan])
    y = np.array([0] * N_SAMPLES + [1] * N_SAMPLES)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classifiers = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (RBF)":           SVC(kernel="rbf", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }

    resultados = {}
    print()
    print("     === Classificação Binária: Real vs. GAN ===")
    for nome, clf in classifiers.items():
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")
        resultados[nome] = scores
        print(f"     {nome:<25s}  {scores.mean():.2%} ± {scores.std():.2%}")

    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    scatter = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y,
                               cmap="RdBu", alpha=0.7, edgecolors="white",
                               linewidths=0.5, s=60)
    plt.colorbar(scatter, ax=axes[0], ticks=[0, 1], label="0=Real | 1=GAN")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)", fontsize=11)
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)", fontsize=11)
    axes[0].set_title("PCA do Espaço Espectral", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    nomes = list(resultados.keys())
    medias = [v.mean() * 100 for v in resultados.values()]
    desvios = [v.std() * 100 for v in resultados.values()]
    bars = axes[1].bar(nomes, medias, color=["steelblue", "mediumseagreen", "darkorange"],
                       alpha=0.85, edgecolor="black", yerr=desvios, capsize=6)
    axes[1].axhline(100, color="tomato", linestyle="--", linewidth=1.5,
                    alpha=0.7, label="100% (ref. artigo)")
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Acurácia (%)", fontsize=12)
    axes[1].set_title("Acurácia por Classificador (5-fold CV)", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, media, desvio in zip(bars, medias, desvios):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + desvio + 1.5,
                     f"{media:.1f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")

    plt.suptitle("Detectabilidade de Deepfakes pelo Espectro de Frequência",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "04_classificacao.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     → Salvo: {path}")

# ---------------------------------------------------------------------------
# Experimento 5 — Sensibilidade ao artefato
# ---------------------------------------------------------------------------

def exp5_sensibilidade():
    print("  [5/5] Análise de sensibilidade ao artefato...")

    amplitudes = [0.01, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20, 0.30]
    acc_vals = []
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, random_state=42)
    X_real = extract_features([natural_image() for _ in range(N_SAMPLES)])

    for amp in amplitudes:
        def gan_amp(a=amp):
            base = natural_image()
            idx = np.indices((SIZE, SIZE)).sum(axis=0) % 2
            return np.clip(base + a * idx, 0, 1)

        X_gan = extract_features([gan_amp() for _ in range(N_SAMPLES)])
        X = np.vstack([X_real, X_gan])
        y = np.array([0] * N_SAMPLES + [1] * N_SAMPLES)
        X_s = scaler.fit_transform(X)
        sc = cross_val_score(clf, X_s, y, cv=3, scoring="accuracy")
        acc_vals.append(sc.mean())

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([a * 100 for a in amplitudes],
            [a * 100 for a in acc_vals],
            "o-", color="steelblue", linewidth=2, markersize=7)
    ax.axhline(50, color="gray", linestyle=":", linewidth=1, label="Baseline aleatório")
    ax.axhline(100, color="tomato", linestyle="--", linewidth=1.5, label="100% (ref. artigo)")
    ax.set_xlabel("Amplitude do Artefato (%)", fontsize=12)
    ax.set_ylabel("Acurácia de Detecção (%)", fontsize=12)
    ax.set_title("Sensibilidade da Detecção vs. Intensidade do Artefato", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "05_sensibilidade.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"     → Salvo: {path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Análise Espectral para Detecção de Deepfakes")
    print("  Durall et al., CVPR 2020 — Reprodução do Grupo")
    print("=" * 60)
    print()

    exp1_espectros_sinteticos()
    exp2_perfis_radiais()
    X_real, X_gan = exp3_comparacao_espectros()
    exp4_classificacao(X_real, X_gan)
    exp5_sensibilidade()

    print()
    print("=" * 60)
    print(f"  Concluído! Resultados em: {RESULTS_DIR}")
    print("=" * 60)
