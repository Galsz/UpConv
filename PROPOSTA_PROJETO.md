# PROPOSTA DE PROJETO — PROCESSAMENTO DIGITAL DE IMAGENS

---

**Instituição:** [Nome da Faculdade]  
**Disciplina:** Processamento Digital de Imagens  
**Semestre:** 2026/1  
**Data de Entrega:** 25/05/2026  
**Apresentação Final:** 01/06/2026  

**Grupo:**
| Nome | RA |
|------|----|
| Geovane Araujo de Lima Silva | 00111884 |
| João Vitor Marinonio de Almeida | — |

---

## 1. Identificação do Artigo

**Título:** Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions

**Autores:** Ricard Durall, Margret Keuper, Janis Keuper

**Ano:** 2020

**Publicação:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020)

**Links:**
- arXiv (versão completa): https://arxiv.org/abs/2003.01826
- PDF direto: https://arxiv.org/pdf/2003.01826
- Código original dos autores: https://github.com/cc-hpc-itwm/UpConv

---

## 2. Descrição do Método

### 2.1 Problema Central

Redes neurais generativas baseadas em convoluções, como as arquiteturas GAN (Generative Adversarial Networks), utilizam operações de **up-convolution** (também chamadas de convolução transposta ou *deconvolução*) para ampliar a resolução espacial das imagens geradas. O artigo demonstra que essas operações introduzem **distorções sistemáticas no espectro de frequência** das imagens sintetizadas.

O efeito é independente da arquitetura GAN específica (DCGAN, ProGAN, WGAN-GP, etc.) e persiste mesmo em modelos treinados com qualidade visual elevada — ou seja, a imagem pode enganar o olho humano mas **revela sua origem sintética no domínio da frequência**.

### 2.2 Fundamentação Teórica

#### Transformada de Fourier 2D

Para uma imagem discreta $f(x,y)$ de dimensão $M \times N$, a DFT 2D é:

$$F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x,y) \cdot e^{-j2\pi\left(\frac{ux}{M}+\frac{vy}{N}\right)}$$

O **espectro de potência** em escala logarítmica (dB) é dado por:

$$P_{dB}(u,v) = 20 \cdot \log\left(|F(u,v)| + \varepsilon\right)$$

#### Integração Azimutal

A integração azimutal reduz o espectro 2D a um **perfil radial 1D**, calculando a média dos coeficientes em anéis concêntricos de raio $r$:

$$\hat{P}(r) = \frac{1}{N_r} \sum_{\{(u,v)\,:\,\lfloor\sqrt{u^2+v^2}\rfloor = r\}} P_{dB}(u,v)$$

Isso torna a representação invariante à rotação e permite comparar imagens de diferentes fontes de forma compacta.

### 2.3 Artefato de Up-Convolution

A operação de up-convolution com *stride* $s$ insere zeros entre os pixels de entrada antes de aplicar o filtro convolucional. Matematicamente, isso é equivalente à multiplicação por uma função periódica de período $1/s$ no domínio da frequência. O resultado é um pico artificial no espectro em $f = 1/s$, produzindo o artefato chamado de **checkerboard** no domínio espacial.

Em imagens naturais, o espectro de potência segue tipicamente uma lei de potência $P(f) \propto 1/f^\alpha$ (decaimento monotônico). Imagens geradas por GAN violam esse padrão com picos em frequências específicas — assinatura detectável pelos métodos do artigo.

### 2.4 Detecção de Deepfakes

Os autores mostram que o perfil espectral 1D (resultado da integração azimutal) é suficiente como *feature* para um **classificador linear** (Regressão Logística) atingir **até 100% de acurácia** em benchmarks públicos (CelebA, FFHQ) na tarefa de distinguir rostos reais de rostos gerados por GAN.

### 2.5 Regularização Espectral (Contribuição Original do Artigo)

Além da análise, o artigo propõe um **termo de perda espectral** adicionado ao objetivo de treinamento do GAN:

$$\mathcal{L}_{total} = \mathcal{L}_{GAN} + \lambda \cdot \mathcal{L}_{freq}$$

Onde $\mathcal{L}_{freq}$ minimiza a diferença entre os perfis espectrais das imagens reais e geradas, forçando o gerador a aprender a distribuição de frequência correta.

---

## 3. Metodologia de Reprodução

### 3.1 Ferramentas Utilizadas

| Ferramenta | Versão | Finalidade |
|------------|--------|------------|
| Python | 3.9+ | Linguagem principal |
| NumPy | 1.23+ | Operações matriciais e FFT (`np.fft.fft2`) |
| OpenCV | 4.6+ | Carregamento e pré-processamento de imagens |
| Matplotlib | 3.5+ | Visualização de espectros e resultados |
| scikit-learn | 1.1+ | Classificadores e avaliação (cross-validation) |
| Jupyter | 6.4+ | Ambiente de experimentação interativo |

### 3.2 O que será Implementado pelo Grupo

O grupo **não usará o código dos autores diretamente**. Nossa implementação é própria, baseada na compreensão do artigo:

1. **`src/radial_profile.py`** — Função de integração azimutal e pipeline FFT completo, implementados do zero com NumPy.

2. **`src/spectral_utils.py`** — Utilitários para carregamento de imagens, cálculo de espectros médios e visualizações comparativas.

3. **`notebooks/01_analise_espectral.ipynb`** — Demonstra o pipeline completo: FFT 2D → espectro de magnitude → perfil radial 1D. Reproduz a Figura 1 do artigo (comparação dos espectros).

4. **`notebooks/02_deteccao_deepfakes.ipynb`** — Reproduz os experimentos de detecção: extração de features espectrais, treinamento de classificadores e avaliação com validação cruzada.

### 3.3 Critério de Validação

Validaremos que nossa reprodução está correta por meio de:

**a) Validação Visual:** nossos gráficos de espectro de potência azimutal devem apresentar o mesmo padrão qualitativo da Figura 1 do artigo — espectro das imagens GAN com excesso de energia em frequências médias/altas em comparação às imagens reais.

**b) Validação Quantitativa:** o classificador linear (Regressão Logística) sobre o perfil espectral 1D deve atingir acurácia alta (idealmente ≥ 90%) na tarefa de distinguir imagens reais de imagens geradas. O artigo reporta 100% em benchmarks com imagens reais; nossa reprodução com dados sintéticos deve aproximar-se desse resultado.

**c) Sanidade do Pipeline FFT:** verificaremos que a FFT da imagem de uma senóide pura produz exatamente os picos nas frequências esperadas, e que o perfil radial é monotonicamente decrescente para imagens naturais suavizadas.

### 3.4 Dataset

Para a entrega final (01/06), utilizaremos:

- **Dataset principal:** Imagens sintéticas geradas programaticamente (imagens naturais simuladas com filtro Gaussiano vs. imagens com artefato checkerboard). Isso garante reprodutibilidade sem dependência de downloads externos.
- **Dataset complementar (se viável):** Subconjunto do [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) ou imagens do conjunto de deepfakes do repositório original dos autores.

---

## 4. Cronograma

| Semana | Período | Atividades |
|--------|---------|------------|
| **Semana 1** | 19/05 – 25/05 | Leitura completa do artigo; implementação dos módulos `src/radial_profile.py` e `src/spectral_utils.py`; validação do pipeline FFT com imagens sintéticas de sanidade. Entrega da **Proposta Final**. |
| **Semana 2** | 26/05 – 01/06 | Finalização dos notebooks (`01_analise_espectral.ipynb` e `02_deteccao_deepfakes.ipynb`); geração e análise de todos os gráficos de resultados; preparação da apresentação final. **Apresentação em 01/06.** |

### Divisão de Tarefas

| Tarefa | Responsável |
|--------|-------------|
| Implementação de `radial_profile.py` | Geovane |
| Implementação de `spectral_utils.py` | João Vitor |
| Notebook 01 (análise espectral) | Geovane |
| Notebook 02 (detecção deepfakes) | João Vitor |
| README e documentação | Ambos |
| Preparação da apresentação | Ambos |

---

## 5. Relevância e Enquadramento na Disciplina

Este projeto se enquadra diretamente na trilha temática **"Análise Forense Avançada"** sugerida pelo professor, especificamente no tópico:

> *"assinaturas de Espectro de Frequência (FFT) em imagens sintéticas ou outras temáticas abordando identificação de deepfakes."*

O método do artigo é intrinsecamente ligado ao conteúdo da disciplina de PDI:
- **Transformada de Fourier 2D:** ferramenta central do domínio da frequência
- **Espectro de potência e fase:** análise de conteúdo espectral de imagens
- **Filtros no domínio da frequência:** compreensão de como filtros (convoluções) afetam o espectro
- **Aplicação forense:** uso de PDI para autenticação e detecção de manipulações

A complexidade é adequada: o artigo é publicado em CVPR (conferência de alto impacto em visão computacional) e apresenta contribuições científicas originais, evitando a categoria de "tutorial simples".

---

## Referências

1. Durall, R., Keuper, M., & Keuper, J. (2020). *Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions*. CVPR 2020. https://arxiv.org/abs/2003.01826

2. Goodfellow, I. et al. (2014). *Generative Adversarial Nets*. NeurIPS 2014.

3. Odena, A., Dumoulin, V., & Olah, C. (2016). *Deconvolution and Checkerboard Artifacts*. Distill. https://distill.pub/2016/deconv-checkerboard/

4. Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.
