# Detecting the Undetected: ML Re-Analysis of Stress-Induced Gait

> **A Machine Learning Re-Analysis of Stress-Induced Gait Alterations During Optical Motion Capture.**

## 📌 Overview

Optical Motion Capture (OMC) is the gold standard for gait analysis, yet the procedure itself—requiring minimal clothing and physical marker placement—can induce psychosocial stress. A foundational study by **Fleischmann et al. (2025)** found that while some participants exhibited an objective physiological stress response (cortisol increase), conventional statistical tests failed to detect any corresponding changes in their gait.

**This project hypothesizes that a subtle, stress-related kinematic signal *does* exist, but requires machine learning to be detected.**

We performed a computational re-analysis of the dataset, employing hypothesis-driven feature engineering and Deep Learning (MoME) to classify "Cortisol Responders" vs. "Non-Responders."

## 🚀 Key Findings

* **Signal Detection:** Our best model (Random Forest) achieved an **AUROC of 0.753** (95% CI: 0.62–0.86), significantly outperforming the random baseline (0.500).
* **Speed Dependency:** The predictive signal is contained almost exclusively in **slow-walking bouts** ( m/s). Fast-walking data ( m/s) acted as noise, likely washing out subtle behavioral "freezing" patterns.
* **Deep Learning vs. Tabular:** Due to the small dataset size (), end-to-end Deep Learning (MoME) failed to generalize. Feature-based tabular models proved far more robust.

## 📂 Repository Structure

```text
├── run_pipeline.py       # Main entry point: Orchestrates the full ML pipeline
├── data.py               # Data loading, merging, and integrity checks
├── preprocess.py         # Sequence building, subject-wise normalization, augmentation
├── features.py           # Feature extraction (Wavelet, Welch, Freezing stats, etc.)
├── tabular.py            # Tabular model implementations (RF, XGBoost, Logistic)
├── sequence_mome.py      # Deep Learning model (Multi-stage Mixture of Movement Experts)
├── validation.py         # Validation helpers (Bootstrap CIs, Threshold tuning)
├── plotting.py           # Visualization tools (ROC, PR, Feature Importance)
└── requirements.txt      # Python dependencies

```

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/gait-stress-detection.git
cd gait-stress-detection

```


2. **Install dependencies:**
It is recommended to use a virtual environment (conda or venv).
```bash
pip install -r requirements.txt

```


*Key dependencies include: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `scipy`, `pywt` (PyWavelets), and `matplotlib`.*
3. **Data Preparation:**
Place the raw data files from the original study in the root directory:
* `kinematics.csv`: Raw joint angles.
* `stride_times.csv`: Timing data for gait cycles.



## 🏃 Usage

Run the complete pipeline using the orchestrator script. This will perform integrity checks, preprocess sequences, extract features, run LOPO cross-validation for all models, and generate plots.

```bash
python run_pipeline.py

```

**Output:**

* Console logs detailing LOPO progress.
* `logs/`: Directory containing ROC curves, confusion matrices, and feature importance plots.
* `summary_metrics_*.csv`: CSV file containing performance metrics for all tested models.
* `participant_features.csv`: The aggregated feature matrix used for tabular modeling.

## 🧠 Methodology

### 1. Preprocessing (`preprocess.py`)

* **Subject-wise Normalization:** Z-scoring per participant to prevent data leakage.
* **Smoothing:** Savitzky-Golay filtering.
* **Derivatives:** Calculation of angular velocity and acceleration.
* **LOPO Safety:** Strict separation of training and testing participants during normalization.

### 2. Feature Engineering (`features.py`)

We engineered ~700 features inspired by biomechanical literature:

* **"Freezing" Statistics:** Fraction of time joint velocity drops below adaptive thresholds (inspired by *Richer et al., 2024*).
* **Frequency Domain:** Welch's method (0-5Hz bandpower, entropy) and Wavelet decomposition (D2-D5, A5) (inspired by *Wang et al., 2024*).
* **Coordination:** Phase lag and cross-correlation between limb pairs (Hip-Knee, Arm-Leg).
* **Segment Aggregates:** Aggregating features by body segment (Legs, Hands, Waist).

### 3. Validation (`validation.py`)

* **Scheme:** Strict **Leave-One-Participant-Out (LOPO)** cross-validation.
* **Tuning:** Nested cross-validation for hyperparameter tuning and threshold optimization (maximizing Balanced Accuracy).
* **Metrics:** AUROC, AUPRC, Balanced Accuracy, Sensitivity, Specificity. 95% Confidence Intervals via participant-level bootstrapping.

## 📊 Results Summary

| Model | Subset | AUROC | Balanced Acc |
| --- | --- | --- | --- |
| Random Baseline | - | 0.500 | 0.500 |
| Logistic Regression | Slow Only | 0.725 | 0.704 |
| **Random Forest** | **Top-20 Slow** | **0.753** | **0.662** |
| XGBoost | Slow Only | 0.625 | 0.611 |
| Logistic Regression | Fast Only | 0.355 | 0.324 |
| Sequence MoME (DL) | All | 0.401 | 0.463 |

> *Note: Models trained on "Fast" walking data performed worse than random guessing, suggesting the stress signal is washed out at higher gait speeds.*

## 📜 References & Acknowledgments

This work is based on the dataset and study design provided by **Fleischmann et al.** Special thanks to the **Machine Learning and Data Analytics Lab (MaD Lab)** at Friedrich-Alexander-Universität Erlangen-Nürnberg.

**Original Study:**

> [1] S. Fleischmann, M. Kurz, R. Richer, et al., "Investigating psychosocial stress arising from marker-based optical motion capture and subsequent effects on gait," *PREPRINT*, July 2025. DOI: [10.21203/rs.3.rs-6957740/v1](https://doi.org/10.21203/rs.3.rs-6957740/v1)

**Key Methodological References:**

> [2] R. Richer et al., "Machine learning-based detection of acute psychosocial stress from body posture and movements," *Scientific Reports*, 2024.
> [3] A. Cătrună et al., "MoME: Estimating psychological traits from gait," *CVPR*, 2024.

---

**Author:** Ahmed Hossam Mohamed Khalil
**Institution:** Friedrich-Alexander-Universität Erlangen-Nürnberg
