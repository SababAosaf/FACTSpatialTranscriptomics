# 🧠 Trainable Clustering Framework for Spatial Transcriptomics

**Authors:**  
Riasat Azim¹*, Sabab Aosaf², Swakkhar Shatabda³, Mohammad Sohel Rahman², and Salekul Islam⁴  
¹ United International University • ² Bangladesh University of Engineering and Technology (BUET)  
³ BRAC University • ⁴ North South University  

📄 Published in *Bioinformatics Advances (Oxford)*, 2025  
📑 Manuscript ID: BIOADV-2025-460  
🔗 [Paper DOI (coming soon)](https://doi.org/)  
💻 [GitHub Repository](https://github.com/SababAosaf/FACTSpatialTranscriptomics)

---

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


![Descriptive alt text](figures/FACT_Methodology.png "Overall Architecture of ACT (A) Initially, Gene count data with associated spatial coordinates are retrieved. A random subset of spatial spots is iteratively sampled, and genes consistently exhibiting high expression across iterations are identified and removed. The resulting filtered and normalized gene count matrix is used to construct a spatial feature map representation for further analysis. (B) An autoencoder is trained by minimizing the reconstruction error between the input and the reconstructed expression profile. (C) A neural network (clustering layer) assigns labels to the latent representation. (D) Using mclust, class labels are assigned to latent representation (an alternative labelling to (C)). Which are then refined using spatial location information. The KL-divergence between the assigned labels and the clustering layer labelling is computed. This divergence is then used to further train the autoencoder along with the clustering layer, enhancing the quality of the final feature representations. (E) Finally, the Mclust algorithm is applied on the learned feature representations to identify distinct spatial domains.")


---

## 🚀 Key Features

- 🧬 Supports **gene expression**, **spatial coordinates**, and **image-derived H&E features**.  
- 🤖 Integrates **autoencoder + MClust** for trainable clustering.  
- 📊 Implements **metaheuristic refinement** (hill-climbing, ASW-based).  
- 🧱 Ensemble mechanism automatically selects the best model by **average silhouette width (ASW)**.  
- 📈 Achieves superior accuracy across multiple benchmark datasets.

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

![Framework Diagram](./docs/architecture.png)

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

```bash
# Clone this repository
git clone https://github.com/SababAosaf/FACTSpatialTranscriptomics.git
cd FACTSpatialTranscriptomics

# Create environment
python -m venv stenv
source stenv/bin/activate  # (or stenv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt
```

---

## 🧬 Usage

### Example: Running the FACT model
```bash
python main.py --method FACT --dataset DLPFC
```

### Example: Running Ensemble framework
```bash
python main.py --method Ensemble --dataset BreastCancer --gpu
```

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

## 📘 Citation

If you use this repository, please cite:

```bibtex
@article{azim2025trainable,
  title={Trainable Clustering Framework for Spatial Transcriptomics},
  author={Azim, Riasat and Aosaf, Sabab and Shatabda, Swakkhar and Rahman, M. Sohel and Islam, Salekul},
  journal={Bioinformatics Advances},
  year={2025},
  publisher={Oxford University Press}
}
```

---

## 🧑‍💻 Contributors

- **Riasat Azim** — Concept, Methodology, Analysis  
- **Sabab Aosaf** — Implementation, Statistical Evaluation  
- **Swakkhar Shatabda** — Method Design, Supervision  
- **M. Sohel Rahman** — Analysis, Review  
- **Salekul Islam** — Supervision, Review  

---

## 🧭 Funding

This research was funded by the  
**Institute for Advanced Research (IAR), United International University**  
Grant Ref. No.: **IAR-02-2023-SE-25**

---

## ⚖️ License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for details.

---

## 📢 Contact

For queries or collaborations, please contact:  
📧 **ri.asim@cse.uiu.ac.bd**  
📧 **sababaosaf@outlook.com**

---

> *“A unified, trainable, and context-aware framework for spatial domain identification in spatial transcriptomics.”*
