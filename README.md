#  Trainable Clustering Framework for Spatial Transcriptomics


## 🧩 Overview

Spatial transcriptomics (ST) enables high-resolution exploration of tissue architecture by integrating gene
expression profiles with spatial information, advancing insights into cellular composition, organization, and
interactions. Among ST applications, spatial domain identification is critical for linking gene expression
patterns to tissue morphology and analyzing the tissue microenvironment. We introduce a trainable clustering
framework that unifies four complementary strategies—ACT, FACT, SCATTER, and their ensemble—into a
cohesive architecture. By coupling autoencoder-driven feature learning with an MClust-assisted clustering layer,
this framework enables joint optimization of representation and cluster assignments through a trainable loss
function. Applied to Human DLPFC, Mouse Brain Anterior, and Human Breast Cancer datasets, the Proposed
framework achieves comparably higher accuracy in most cases while reliably identifying spatial domains and
preserving complex tissue architecture

Spatial transcriptomics (ST) allows researchers to integrate **gene expression profiles** with **spatial tissue information**, enabling high-resolution exploration of tissue architecture.  
This repository implements a **trainable clustering framework** that unifies four complementary methods:

- **ACT** — Autoencoder-Clustering framework  
- **FACT** — Filtered Autoencoder-Clustering framework  
- **SCATTER** — Spot-based filtration & autoencoding  
- **ENSEMBLE** — Adaptive combination of the three methods  

The framework couples **autoencoder-driven feature learning** with an **MClust-based clustering layer**, enabling **joint optimization of latent representations and cluster assignments** via a trainable loss function.



---

## 🚀 Key Features

-  Supports **gene expression**, **spatial coordinates**, and **image-derived H&E features**.  
-  Integrates **autoencoder + MClust** for trainable clustering.  
-  Implements **metaheuristic refinement** (hill-climbing, ASW-based).  
-  Ensemble mechanism automatically selects the best model by **average silhouette width (ASW)**.  
-  Achieves superior accuracy across multiple benchmark datasets.

---

## 📚 Datasets

| Dataset | Platform | Spots | Slides |
|----------|-----------|--------|---------|
| **Human DLPFC** | 10x Visium | 3,460–4,789 | 12 |
| **Mouse Brain Anterior** | 10x Visium | 2,695 | 1 |
| **Human Breast Cancer** | 10x Visium | 3,798 | 1 |

Public datasets are available from **10x Genomics** and **Nature Neuroscience / Communications** repositories.

---

## 🧠 Framework Architecture

The workflow consists of five core stages:

1. **Gene Filtration** – Remove highly frequent genes and retain high-variance ones using Seurat v3.  
2. **Normalization** – Apply library size scaling, log normalization, z-scoring, and outlier clipping.  
3. **Latent Representation** – Train an **autoencoder** to learn compact latent features.  
4. **Clustering Layer** – Integrate **MClust** and neural clustering; optimize using **KL-divergence loss**.  
5. **Spatial Refinement** – Smooth cluster labels using neighboring spatial consistency.  


![Descriptive alt text](figures/FACT_Methodology.png "Overall Architecture of ACT (A) Initially, Gene count data with associated spatial coordinates are retrieved. A random subset of spatial spots is iteratively sampled, and genes consistently exhibiting high expression across iterations are identified and removed. The resulting filtered and normalized gene count matrix is used to construct a spatial feature map representation for further analysis. (B) An autoencoder is trained by minimizing the reconstruction error between the input and the reconstructed expression profile. (C) A neural network (clustering layer) assigns labels to the latent representation. (D) Using mclust, class labels are assigned to latent representation (an alternative labelling to (C)). Which are then refined using spatial location information. The KL-divergence between the assigned labels and the clustering layer labelling is computed. This divergence is then used to further train the autoencoder along with the clustering layer, enhancing the quality of the final feature representations. (E) Finally, the Mclust algorithm is applied on the learned feature representations to identify distinct spatial domains.")

Description: Overall Architecture of ACT (A) Initially, Gene count data with associated spatial coordinates are retrieved. A random subset of spatial spots is iteratively sampled, and genes consistently exhibiting high expression across iterations are identified and removed. The resulting filtered and normalized gene count matrix is used to construct a spatial feature map representation for further analysis. (B) An autoencoder is trained by minimizing the reconstruction error between the input and the reconstructed expression profile. (C) A neural network (clustering layer) assigns labels to the latent representation. (D) Using mclust, class labels are assigned to latent representation (an alternative labelling to (C)). Which are then refined using spatial location information. The KL-divergence between the assigned labels and the clustering layer labelling is computed. This divergence is then used to further train the autoencoder along with the clustering layer, enhancing the quality of the final feature representations. (E) Finally, the Mclust algorithm is applied on the learned feature representations to identify distinct spatial domains.

---

## 🧪 Performance Summary

| Dataset | Best Method | Avg. ARI | Median ARI | Notes |
|----------|--------------|----------|-------------|--------|
| DLPFC | Ensemble | **0.50** | 0.515 | Highest accuracy on 6/12 slides |
| Mouse Brain Anterior | Scatter | 0.34 | 0.33 | High NMI = 0.61 |
| Human Breast Cancer | ACT / Ensemble | **0.61** | 0.60 | Outperforms DeepST, GraphST, STAGATE |

Additional metrics used:
- **Adjusted Rand Index (ARI)**
- **Normalized Mutual Information (NMI)**
- **Moran’s I & Geary’s C**
- **PAS & CHAOS scores**
- **Completeness & Homogeneity**

---

## ⚙️ Installation

1. The project requires NVIDIA GPU. Install Driver Version: 581.42, CUDA Version: 13.0 and Cudnn.
2. Intall R (4.3.2)
3. Install Anaconda (Python version 3.11)
4. Create a virtual environment in Anaconda for this project.
5. Clone this repository.
```bash
git clone https://github.com/SababAosaf/FACTSpatialTranscriptomics.git
cd FACTSpatialTranscriptomics
```
6. Open the FACTSpatialTranscriptomics project in an IDE (preferably PyCharm).
4. In IDE, select the python.exe of the virtual environment (of Anaconda) as a python interpreter.
5. Intall all the packages given in requirements.txt (some packages may need to be installed via anaconda navigator).


---

## 🧬 Usage

In main file scatter.scatter('DLPFC'), FACT.FACT('DLPFC') and ACT.ACT('DLPFC') runs the clustering methods.  Methods will run for 'DLPFC'. Any compatible dataset can be given in place of it.


### Arguments
| Flag | Description |
|------|--------------|
| `--method` | Select model: `ACT`, `FACT`, `SCATTER`, or `Ensemble` |
| `--dataset` | Choose dataset name (DLPFC / MouseBrain / BreastCancer) |
| `--gpu` | (Optional) Enable GPU acceleration |
| `--save-path` | Specify output directory for results |

Outputs include:
- Cluster assignments  
- ARI/NMI metrics  
- Spatial domain plots  

---

## 📊 Benchmark Comparison

Compared with state-of-the-art methods:

| Method | Median ARI (DLPFC) |
|--------|---------------------|
| **Proposed Ensemble** | **0.515** |
| STAGATE | 0.510 |
| GraphST | 0.485 |
| DeepST | 0.490 |
| conST | 0.390 |
| SpaceFlow | 0.260 |

---

## 📢 Contact

For queries or collaborations, please contact:  
📧 **sababaosaf19@gmail.com**

---

> *“A unified, trainable, and context-aware framework for spatial domain identification in spatial transcriptomics.”*
