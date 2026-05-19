"""
Perfil Radial - Integração Azimutal do Espectro de Potência
============================================================
Autores: Geovane Araujo de Lima Silva (RA: 00111884)
         João Vitor Marinonio de Almeida

Disciplina: Processamento Digital de Imagens
Implementação própria baseada no método descrito em:
  Durall et al., "Watch your Up-Convolution: CNN Based Generative Deep Neural
  Networks are Failing to Reproduce Spectral Distributions", CVPR 2020.
  https://arxiv.org/abs/2003.01826
"""

import numpy as np


def azimuthal_average(image: np.ndarray, center: list = None) -> np.ndarray:
    """
    Calcula o perfil radial médio azimutal de uma imagem 2D.

    A integração azimutal reduz o espectro de potência 2D a um perfil 1D
    agrupando os coeficientes por distância radial ao centro do espectro.
    Isso permite comparar a distribuição espectral de imagens reais vs. geradas
    de forma independente da rotação.

    Parâmetros
    ----------
    image : np.ndarray
        Imagem 2D (espectro de magnitude ou power spectrum).
    center : list, opcional
        Coordenadas [x, y] do pixel central. Se None, usa o centro geométrico
        da imagem.

    Retorna
    -------
    np.ndarray
        Perfil radial 1D representando a média azimutal por bin de raio.

    Referência
    ----------
    Adaptado de: https://www.astrobetter.com/blog/2010/03/03/
                 fourier-transforms-of-images-in-python/
    """
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([
            (x.max() - x.min()) / 2.0,
            (y.max() - y.min()) / 2.0
        ])

    # Distância euclidiana de cada pixel ao centro
    r = np.hypot(x - center[0], y - center[1])

    # Ordena os raios e obtém os índices de ordenação
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Parte inteira dos raios (tamanho do bin = 1 pixel)
    r_int = r_sorted.astype(int)

    # Localiza as transições entre bins de raio
    delta_r = r_int[1:] - r_int[:-1]
    rind = np.where(delta_r)[0]
    nr = rind[1:] - rind[:-1]  # quantidade de pixels por bin

    # Soma acumulada para calcular a média por bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_profile = tbin / nr

    return radial_profile


def compute_power_spectrum_1d(image_gray: np.ndarray,
                               epsilon: float = 1e-8) -> np.ndarray:
    """
    Calcula o espectro de potência 1D de uma imagem em escala de cinza.

    Aplica a FFT 2D, centraliza o espectro (fftshift), calcula a magnitude
    em escala logarítmica e realiza a integração azimutal.

    Parâmetros
    ----------
    image_gray : np.ndarray
        Imagem 2D em escala de cinza (valores float).
    epsilon : float
        Valor mínimo para evitar log(0).

    Retorna
    -------
    np.ndarray
        Perfil espectral 1D normalizado em [0, 1].
    """
    # Transformada de Fourier 2D
    fft = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(fft)
    fshift += epsilon

    # Espectro de magnitude em dB
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Integração azimutal → perfil 1D
    psd1d = azimuthal_average(magnitude_spectrum)

    # Normalização min-max
    psd1d = (psd1d - psd1d.min()) / (psd1d.max() - psd1d.min() + epsilon)

    return psd1d


def rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    """
    Converte imagem RGB para escala de cinza usando ponderação ITU-R BT.601.

    Parâmetros
    ----------
    image_rgb : np.ndarray
        Imagem RGB com shape (H, W, 3).

    Retorna
    -------
    np.ndarray
        Imagem em escala de cinza com shape (H, W).
    """
    r, g, b = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    return 0.2989 * r + 0.5870 * g + 0.1140 * b
