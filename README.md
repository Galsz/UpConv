# Análise Espectral de Frequência para Detecção de Deepfakes

**Disciplina:** Processamento Digital de Imagens  
**Semestre:** 2026/1  
**Grupo:**
- Geovane Araujo de Lima Silva — RA: 00111884
- João Vitor Marinonio de Almeida

---

## Sobre o Projeto

Este repositório contém a reprodução dos experimentos do artigo científico:

> **Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions**  
> Ricard Durall, Margret Keuper, Janis Keuper  
> *CVPR 2020* — [https://arxiv.org/abs/2003.01826](https://arxiv.org/abs/2003.01826)

O trabalho demonstra que redes neurais generativas (GANs) introduzem **distorções sistemáticas no espectro de frequência** das imagens geradas. Essas distorções surgem das operações de *up-convolution* (convolução transposta) e são invisíveis no domínio espacial, mas claramente detectáveis no domínio da frequência via **Transformada de Fourier 2D (FFT)**.

---

## Problema Abordado

Imagens sintéticas geradas por GANs (como DCGAN, WGAN, ProGAN) apresentam um padrão espectral característico: **energia excessiva em frequências médias e altas**, resultante dos artefatos periódicos das operações de upsampling com stride. Este fenômeno pode ser explorado para **detectar deepfakes automaticamente** sem treinamento de classificadores complexos.

![Comparação espectral](results/04_comparacao_espectros.png)

---

## Método Reproduzido

### Pipeline de Análise

```
Imagem RGB
    │
    ▼
Conversão para Escala de Cinza (ITU-R BT.601)
    │
    ▼
FFT 2D  →  fftshift  →  Espectro de Magnitude (dB)
    │
    ▼
Integração Azimutal  →  Perfil Radial 1D (normalizado)
    │
    ▼
Classificador Linear (Regressão Logística / SVM)
    │
    ▼
Real ou Gerado por GAN?
```

### Componentes Principais

| Arquivo | Descrição |
|---------|-----------|
| `src/radial_profile.py` | Implementação da integração azimutal e pipeline FFT |
| `src/spectral_utils.py` | Utilitários de carregamento, processamento e visualização |
| `notebooks/01_analise_espectral.ipynb` | Análise espectral fundamental e reprodução da Figura 1 do artigo |
| `notebooks/02_deteccao_deepfakes.ipynb` | Detecção de deepfakes e comparação de classificadores |

---

## Estrutura de Pastas

```
UpConv/
├── notebooks/
│   ├── 01_analise_espectral.ipynb     # Análise FFT e perfil azimutal
│   └── 02_deteccao_deepfakes.ipynb    # Detecção de imagens geradas
├── src/
│   ├── radial_profile.py              # Integração azimutal (implementação própria)
│   └── spectral_utils.py             # Utilitários de análise espectral
├── data/
│   └── sample_images/                # Imagens de teste (opcional)
├── imgs/                             # Figuras do artigo de referência
├── results/                          # Figuras geradas pelos experimentos
└── README.md
```

---

## Como Executar

### 1. Instalar dependências

```bash
pip install numpy opencv-python matplotlib scikit-learn jupyter
```

### 2. Executar os notebooks

```bash
cd notebooks
jupyter notebook
```

Abrir e executar em ordem:
1. `01_analise_espectral.ipynb`
2. `02_deteccao_deepfakes.ipynb`

---

## Experimentos Reproduzidos

### Experimento 1 — Visualização do Espectro FFT
Demonstra como diferentes tipos de imagem (natural, senoidal, ruído, grade) produzem espectros FFT distintos.

### Experimento 2 — Perfil Radial Azimutal
Calcula e compara os perfis espectrais 1D de imagens reais vs. geradas por GAN.

### Experimento 3 — Detecção Automática
Avalia a acurácia de três classificadores (Regressão Logística, SVM, Random Forest) usando apenas o perfil espectral como feature.

### Experimento 4 — Análise de Sensibilidade
Investiga como a intensidade do artefato de up-convolution afeta a detectabilidade.

---

## Resultados

Os experimentos confirmam as conclusões do artigo original:

- **Classificadores lineares** sobre o perfil espectral 1D são suficientes para detectar imagens geradas por GAN com alta acurácia.
- A **integração azimutal** reduz eficientemente o espectro 2D a um vetor 1D discriminativo.
- Mesmo **artefatos sutis** (amplitude < 5%) são detectáveis no domínio da frequência.

---

## Referência

```bibtex
@InProceedings{durall2020upconv,
  title     = {Watch Your Up-Convolution: CNN Based Generative Deep Neural Networks
               are Failing to Reproduce Spectral Distributions},
  author    = {Durall, Ricard and Keuper, Margret and Keuper, Janis},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
               and Pattern Recognition (CVPR)},
  year      = {2020},
  eprint    = {2003.01826},
  url       = {https://arxiv.org/abs/2003.01826}
}
```

---

## Licença

Este projeto é de uso acadêmico, desenvolvido como reprodução de método científico para a disciplina de Processamento Digital de Imagens. O artigo original é de autoria de Durall, Keuper e Keuper (2020).
