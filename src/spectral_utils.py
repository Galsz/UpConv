"""
Utilitários de Análise Espectral
==================================
Autores: Geovane Araujo de Lima Silva (RA: 00111884)
         João Vitor Marinonio de Almeida

Disciplina: Processamento Digital de Imagens
Implementação própria baseada no método descrito em:
  Durall et al., "Watch your Up-Convolution: CNN Based Generative Deep Neural
  Networks are Failing to Reproduce Spectral Distributions", CVPR 2020.
  https://arxiv.org/abs/2003.01826
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from radial_profile import compute_power_spectrum_1d, rgb_to_gray


# ---------------------------------------------------------------------------
# Carregamento e pré-processamento de imagens
# ---------------------------------------------------------------------------

def load_image(path: str, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Carrega uma imagem, converte BGR→RGB e redimensiona.

    Parâmetros
    ----------
    path : str
        Caminho para o arquivo de imagem.
    size : tuple
        Dimensão alvo (largura, altura).

    Retorna
    -------
    np.ndarray
        Imagem RGB normalizada em [0, 1] com shape (H, W, 3).
    """
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, size)
    return img_resized.astype(np.float32) / 255.0


def load_images_from_dir(directory: str,
                          size: Tuple[int, int] = (128, 128),
                          max_count: int = 200) -> List[np.ndarray]:
    """
    Carrega todas as imagens de um diretório (formatos .jpg, .png, .jpeg).

    Parâmetros
    ----------
    directory : str
        Caminho do diretório.
    size : tuple
        Dimensão alvo de redimensionamento.
    max_count : int
        Número máximo de imagens a carregar.

    Retorna
    -------
    list de np.ndarray
        Lista de imagens RGB normalizadas.
    """
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = [
        p for p in Path(directory).iterdir()
        if p.suffix.lower() in extensions
    ][:max_count]

    images = []
    for p in paths:
        try:
            images.append(load_image(str(p), size))
        except FileNotFoundError:
            continue

    return images


# ---------------------------------------------------------------------------
# Cálculo do espectro médio de um conjunto de imagens
# ---------------------------------------------------------------------------

def compute_mean_spectrum(images: List[np.ndarray],
                           n_bins: int = 88) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcula a média e o desvio padrão do espectro de potência 1D para um
    conjunto de imagens.

    Parâmetros
    ----------
    images : list de np.ndarray
        Imagens RGB normalizadas.
    n_bins : int
        Número de bins espectrais a considerar (determina o comprimento do
        vetor de saída).

    Retorna
    -------
    mean_psd : np.ndarray
        Espectro médio (n_bins,).
    std_psd : np.ndarray
        Desvio padrão do espectro (n_bins,).
    """
    spectra = []
    for img in images:
        gray = rgb_to_gray(img)
        psd = compute_power_spectrum_1d(gray)
        if len(psd) >= n_bins:
            spectra.append(psd[:n_bins])

    if not spectra:
        raise ValueError("Nenhuma imagem produziu espectro com bins suficientes.")

    stack = np.stack(spectra, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def plot_spectrum_comparison(mean_real: np.ndarray,
                              std_real: np.ndarray,
                              mean_fake: np.ndarray,
                              std_fake: np.ndarray,
                              label_real: str = "Imagens Reais",
                              label_fake: str = "Imagens Geradas (GAN)",
                              title: str = "Comparação do Espectro de Potência Azimutal",
                              save_path: str = None) -> None:
    """
    Plota a comparação entre o espectro de potência médio de imagens reais e
    geradas por GAN, com banda de ±1 desvio padrão.

    Parâmetros
    ----------
    mean_real, std_real : np.ndarray
        Média e desvio padrão do espectro das imagens reais.
    mean_fake, std_fake : np.ndarray
        Média e desvio padrão do espectro das imagens geradas.
    label_real, label_fake : str
        Rótulos para a legenda.
    title : str
        Título do gráfico.
    save_path : str, opcional
        Se fornecido, salva a figura nesse caminho.
    """
    x = np.arange(len(mean_real))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, mean_real, color="steelblue", label=label_real, linewidth=2)
    ax.fill_between(x,
                    mean_real - std_real,
                    mean_real + std_real,
                    alpha=0.3, color="steelblue")

    ax.plot(x, mean_fake, color="tomato", label=label_fake, linewidth=2,
            linestyle="--")
    ax.fill_between(x,
                    mean_fake - std_fake,
                    mean_fake + std_fake,
                    alpha=0.3, color="tomato")

    ax.set_xlabel("Frequência Radial (bin)", fontsize=12)
    ax.set_ylabel("Potência Normalizada", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura salva em: {save_path}")

    plt.show()


def plot_fft_visualization(image: np.ndarray,
                            title: str = "Análise FFT",
                            save_path: str = None) -> None:
    """
    Visualiza uma imagem junto com seu espectro de magnitude FFT (escala log).

    Parâmetros
    ----------
    image : np.ndarray
        Imagem RGB normalizada.
    title : str
        Título da figura.
    save_path : str, opcional
        Caminho para salvar a figura.
    """
    gray = rgb_to_gray(image)

    fft = np.fft.fft2(gray)
    fshift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(image)
    axes[0].set_title("Imagem Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("Escala de Cinza")
    axes[1].axis("off")

    im = axes[2].imshow(magnitude, cmap="inferno")
    axes[2].set_title("Espectro de Magnitude FFT (dB)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figura salva em: {save_path}")

    plt.show()
