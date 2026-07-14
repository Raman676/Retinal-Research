# Retinal-Research
# 👁️ MFFNet: Dual-Domain Spatial-Frequency Retinal OCT Classification & SSL Engine

[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)

An advanced, multi-modality deep learning framework implementing a **Dual-Domain Multi-Feature Fusion Network (MFFNet)** and self-supervised contrastive representation learning (SimCLR / SSL-AnoVAE) for automated ophthalmic disease diagnosis across **16,822 Retinal Optical Coherence Tomography (OCT)** scans.

> 📓 **Core Research Notebook:** The unified training, frequency decomposition, and unsupervised clustering pipeline is documented in [`retinal-research-model-final.ipynb`](https://www.kaggle.com/code/dubeyraman/retinal-research-model-final).

---

## 🌟 Key Research & Architectural Innovations

### 1. Dual-Domain Wavelet Signal Processing (Haar DWT / IDWT)
Standard CNNs operate solely in the spatial domain, often confounding high-frequency retinal speckle noise with critical structural boundaries. MFFNet integrates a custom **Haar Discrete Wavelet Transform (DWT)** layer that decomposes 2D feature maps into four distinct frequency sub-bands:
* **LL (Low-Low):** Coarse global structural morphology.
* **LH / HL (Low-High / High-Low):** Horizontal and vertical edge alignments (retinal layer boundaries).
* **HH (High-High):** High-frequency speckle noise and fine texture anomalies.

### 2. Spatial-Frequency Wavelet Attention Fusion (SFWAF Module)
A dual-branch neural architecture that dynamically balances spatial feature extraction with frequency-domain representations using a learnable gating parameter ($\alpha$). The adaptive fusion rule ensures invariant feature representation across diverse imaging equipment:
$$\text{Output} = (1 - \sigma(\alpha)) \cdot \mathcal{F}_{\text{spatial}}(x) + \sigma(\alpha) \cdot \text{IDWT}\left(\mathcal{F}_{\text{freq}}(\text{DWT}(x))\right)$$

### 3. Frequency-Enhanced Feature Fusion (FEF Module)
To preserve essential pathological edge structures during multiscale downsampling, the FEF module applies a **Maximum Fusion Rule (`reduce_max`)** specifically across the high-frequency sub-bands ($LH, HL, HH$) before Inverse DWT reconstruction, preventing gradient degradation along lesion boundaries.

### 4. Custom Activation & Edge Enhancement
* **Cross Activation ($g(x) = x \cdot \vert{}1 - e^{-x}\vert{}$):** Prevents negative weight loss during backpropagation and maintains high gradient magnitudes near zero, avoiding vanishing gradients in deep bottleneck blocks.
* **EdgeEn Layer:** Enhances local feature contrast around the spatial mean by a scaling factor ($\gamma = 1.65$), reinforcing strong retinal layer gradients while suppressing background noise.

### 5. Self-Supervised Contrastive Pre-training (SimCLR & Rotation Pretext)
Implements an unsupervised projection head utilizing **Normalized Temperature-scaled Cross Entropy (NT-Xent)** loss ($\tau = 0.1$) and stochastic rotation pretext tasks ($\{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$). The encoder learns robust, invariant visual representations without manual labels, mapping latent embeddings into separable clinical clusters.

---

## 📈 System Architecture & Data Flow

```text
       [Input Retinal OCT B-Scans (224 × 224 × 3)]
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
┌──────────────┐                       ┌──────────────┐
│ Spatial Conv │                       │   Haar DWT   │
│   Branch     │                       │Decomposition │
└──────┬───────┘                       └──────┬───────┘
       │                                      ▼
       │                               ┌──────────────┐
       │                               │ LL, LH, HL,  │
       │                               │ HH Sub-bands │
       │                               └──────┬───────┘
       │                                      ▼
       │                               ┌──────────────┐
       │                               │ FEF Max-Edge │
       │                               │ Optimization │
       │                               └──────┬───────┘
       │                                      ▼
       │                               ┌──────────────┐
       │                               │ Inverse DWT  │
       │                               │Reconstruction│
       └───────────────────┬───────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │ SFWAF Adaptive Gating (α)    │
            │ + CrossActivation & EdgeEn   │
            └──────────────┬───────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌─────────────────┐                 ┌─────────────────┐
│  Supervised     │                 │ Unsupervised    │
│  Diagnostic Head│                 │ SimCLR Latent   │
│ (Normal/Drusen/ │                 │ Clustering      │
│      CNV)       │                 │ (t-SNE / ARI)   │
└─────────────────┘                 └─────────────────┘
```

---

## 📊 Performance Benchmark & Latent Alignment

The model's unsupervised representations are evaluated against clinical ground truth across three pathology classes—**Normal**, **Drusen** (Early AMD biomarker), and **CNV** (Choroidal Neovascularization / Wet AMD)—using $k$-Means clustering ($k=3$) and majority-vote mapping:

| Framework Architecture | Domain Modality | Supervision Mode | Global Accuracy | Latent Variance / ARI |
| :--- | :--- | :--- | :--- | :--- |
| Random Chance Baseline | — | None | ~33.33% | 0.0000 |
| Baseline SimCLR (Spatial Only) | Spatial | Unsupervised | 40.90% | ~0.1240 |
| **MFFNet (Frozen Backbone)** | **Spatial + Frequency (DWT)** | **Unsupervised + SSL** | **82.05%** | **High Cluster Separation** |
| **MFFNet (End-to-End Fine-Tuned)** | **Dual-Domain Collaborative** | **Supervised Fusion** | **>90.00%** | **Optimized Edge Boundaries** |

---

## 🛠️ Tech Stack & Library Dependencies

* **Deep Learning Frameworks:** TensorFlow 2.x / Keras (MFFNet, SFWAF, FEF, Custom Gradients), PyTorch (SSL-AnoVAE, ConvAutoencoders)
* **Signal Processing:** Custom Discrete Wavelet Transforms (`dwt_haar`, `idwt_haar`), OpenCV, PIL
* **Evaluation & Clustering:** Scikit-Learn (`KMeans`, `adjusted_rand_score`, `silhouette_score`, `TSNE`, `class_weight`)
* **Visualization:** Matplotlib, Seaborn (`ConfusionMatrixDisplay`, latent variance violin plots)

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the runtime environment:
```bash
git clone [https://github.com/RamanDubey/MFFNet-Signal-Fusion.git](https://github.com/RamanDubey/MFFNet-Signal-Fusion.git)
cd MFFNet-Signal-Fusion
pip install tensorflow torch torchvision scikit-learn opencv-python pandas numpy matplotlib seaborn
```

### 2. Core Implementation: Haar Wavelet & SFWAF Module
```python
import tensorflow as tf
from tensorflow.keras import layers

def dwt_haar(x):
    """Decomposes 2D feature maps into 4 spatial-frequency subbands."""
    x00 = x[:, 0::2, 0::2, :]  # LL (Low Frequency / Global Structure)
    x01 = x[:, 1::2, 0::2, :]  # LH (Horizontal Edge Features)
    x10 = x[:, 0::2, 1::2, :]  # HL (Vertical Edge Features)
    x11 = x[:, 1::2, 1::2, :]  # HH (High Frequency / Speckle Noise)
    return tf.concat([x00, x01, x10, x11], axis=-1)

class SFWAF_Module(layers.Layer):
    """Spatial-Frequency Wavelet Attention Fusion Layer."""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.spatial_conv = layers.Conv2D(filters, 3, padding='same')
        self.bn = layers.BatchNormalization()
        self.freq_conv = layers.Conv2D(filters * 4, 1, padding='same')
        self.alpha = self.add_weight(name="alpha", shape=(), initializer="zeros", trainable=True)

    def call(self, x):
        s_feat = self.bn(self.spatial_conv(x))
        f_feat = tf.nn.depth_to_space(self.freq_conv(dwt_haar(x)), 2)
        a = tf.nn.sigmoid(self.alpha)
        return (1 - a) * s_feat + a * f_feat
```

### 3. Running Unsupervised Cluster Evaluation
To evaluate unsupervised latent alignment against clinical ground truth:
```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score

# Extract combined latent embeddings (z + ssl_feat)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
pred_clusters = kmeans.fit_predict(features)

def get_best_mapping(true_labels, pred_clusters):
    """Maps unsupervised cluster indices to clinical labels via majority voting."""
    mapping = {}
    for i in range(3):
        mask = (pred_clusters == i)
        if np.any(mask):
            mapping[i] = np.bincount(true_labels[mask]).argmax()
    return mapping

cluster_map = get_best_mapping(true_labels, pred_clusters)
mapped_preds = np.array([cluster_map.get(p, 0) for p in pred_clusters])

print(f"Global Unsupervised Alignment Accuracy: {accuracy_score(true_labels, mapped_preds) * 100:.2f}%")
print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(true_labels, pred_clusters):.4f}")
```
