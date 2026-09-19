
# PINN-XGBoost: Physics-Informed Neural Network with XGBoost Residual Correction for Nuclear Separation Energy Predictions

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid machine-learning framework for predicting one-neutron (\(S_n\)) and one-proton (\(S_p\)) separation energies across the nuclear chart, combining a **Physics-Informed Neural Network (PINN)** with **XGBoost residual correction**. The model is trained on the **AME2020** atomic mass evaluation and incorporates the **Liquid Drop Model (LDM)** as an explicit physics constraint.

---

## 📊 Key Results

| Quantity | RMSE (keV) | R² |
|---|---|---|
| \(S_n\) | **333.5** | **0.9868** |
| \(S_p\) | **334.1** | **0.9945** |

These results represent a **~50% reduction in RMSE** compared to standard nuclear mass models (FRDM2012, WS4) and **>40% improvement** over standalone machine-learning approaches.

---

## 🧠 Method Overview

The framework consists of two stages:

1. **Physics-Informed Neural Network (PINN)**
   - 4 hidden layers × 128 neurons, LeakyReLU activation, dropout 0.1
   - Residual connections between input/first hidden layer and first/second hidden layers
   - 12 physics-motivated input features (see Table 1 of the manuscript)
   - Loss = MSE + λ₁·L_physics + λ₂·L_smooth
     - `L_physics`: MSE between prediction and LDM separation energy (λ₁ = 0.01)
     - `L_smooth`: smoothness along isotopic chains (λ₂ = 0.005)

2. **XGBoost Residual Correction**
   - XGBoost regressor trained on the PINN residuals: r_i = y_i^exp − y_i^PINN
   - Final prediction: y_i^final = y_i^PINN + XGBoost(r_i)

---

## ⚙️ Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- XGBoost ≥ 1.7
- scikit-learn ≥ 1.2
- NumPy, pandas, SciPy, Matplotlib, Seaborn, Optuna

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/PINN-XGBoost-Nuclear-Separation-Energies.git
cd PINN-XGBoost-Nuclear-Separation-Energies

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`

```
numpy>=1.24
pandas>=2.0
torch>=2.0
xgboost>=1.7
scikit-learn>=1.2
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
optuna>=3.0
shap>=0.42
```

---

## 📥 Data

The dataset used in this study is derived from the **Atomic Mass Evaluation 2020 (AME2020)**:

> M. Wang, W.J. Huang, F.G. Kondev, G. Audi, S. Naimi,
> *The AME2020 atomic mass evaluation (II). Tables, graphs and references*,
> Chinese Physics C **45**, 030003 (2021).

The raw AME2020 data are publicly available at:
- https://www-nds.iaea.org/amdc/
- https://amdc.impcas.ac.cn/

### Preparing `ame2020_predictions.csv`

The pipeline expects a CSV file with at least the following columns:

| Column | Description |
|---|---|
| `N` | Neutron number |
| `Z` | Proton number |
| `A` | Mass number |
| `S_n` | One-neutron separation energy (keV) |
| `S_p` | One-proton separation energy (keV) |

After filtering (experimental entries only, Z > 0, N > 0, A > 0, physically reasonable S_n/S_p ranges), the dataset contains **2209 nuclei** with Z = 1–107 and N = 1–157.

---

## 🚀 Usage

### Quick Start

```bash
python mainprog.py
```

This will:
1. Load and clean the AME2020 data.
2. Engineer the 12 physics-inspired features.
3. Train Baseline MLP, PINN, and PINN + XGBoost models.
4. Perform uncertainty quantification via a 10-member ensemble.
5. Compute SHAP/permutation feature importance.
6. Generate all publication-quality figures (600 DPI, Times New Roman) in `results/`.
7. Write a summary report to `results/report.txt`.

### Configuration

All hyperparameters are centralized in the `Config` dataclass at the top of `mainprog.py`:

```python
@dataclass
class Config:
    data_path: str = 'ame2020_predictions.csv'
    test_size: float = 0.2
    random_state: int = 42
    input_dim: int = 12
    hidden_dim: int = 128
    hidden_layers: int = 4
    dropout: float = 0.1
    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 1e-3
    physics_weight: float = 0.01
    ...
```

Edit these values directly to reproduce variations of the experiment.

---

## 🔬 Reproducibility

- All random seeds are fixed via `set_seeds(config.seed)` for NumPy, PyTorch, and Python's `random`.
- `torch.backends.cudnn.deterministic = True` and `benchmark = False` are set.
- Single train/test split (80/20, stratified over A) is used to avoid data leakage.
- Leave-one-isotope-chain-out cross-validation is additionally performed for robustness.
- All results in the manuscript can be reproduced by running `python mainprog.py` on the prepared `ame2020_predictions.csv`.

---

## 🖼️ Output Figures

| Figure | Description |
|---|---|
| `Figure1_Performance_Ablation.png` | Performance comparison with WS4, FRDM2012, and ablation study |
| `Figure2_Parity_Plot_S_n.png` | Parity plot for \(S_n\) (RMSE = 333.5 keV, R² = 0.9868) |
| `Figure2_Parity_Plot_S_p.png` | Parity plot for \(S_p\) (RMSE = 334.1 keV, R² = 0.9945) |
| `Figure3_Heatmaps.png` | Predicted \(S_n\) and \(S_p\) across the nuclear chart |

---

## 📖 Citation

If you use this code or the trained models in your research, please cite:

```bibtex
@article{Khalili2026PINN,
  title   = {PINN-XGBoost: A Novel Hybrid Approach for Nuclear Separation Energy Predictions},
  author  = {Khalili, H. and ...},
  journal = {Communications Physics},
  year    = {2026},
  note    = {Manuscript under review}
}
```

> Update the author list and journal information once the paper is accepted.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- The AME2020 collaboration for maintaining the atomic mass evaluation.
- The open-source communities behind PyTorch, XGBoost, scikit-learn, and SHAP.

---

## 📬 Contact

For questions, bug reports, or collaboration inquiries, please open an issue or contact:

**Hassan Khalili** — corresponding author
Arak University
Email: h-khalili@araku.ac.ir
**Mahdi AzadMarzabadi** — Software and Coauthor
Arak University
azadmahdi19@gmail.com
