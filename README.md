# 🌙 LUNA-REG: Automated Multi-Sensor Lunar Image Registration Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

**LUNA-REG** is a high-precision satellite image registration pipeline built for lunar spatial data (such as Chandrayaan-2 TMC, OHRC, and IIRS sensors). It handles cross-modal alignment using a two-stage hybrid approach: coarse keypoint matching via AKAZE/ORB + RANSAC, followed by sub-pixel fine optimization using Normalized Mutual Information (NMI).

---

## 🚀 Key Features

* **Two-Stage Registration:** Coarse global alignment (RANSAC homography) + sub-pixel fine optimization (L-BFGS-B NMI).
* **Preprocessing Pipeline:** Automatic CLAHE contrast enhancement and Laplacian edge preservation.
* **Spatial Integrity Guardrails:** RANSAC inlier verification and spatial entropy distribution scoring.
* **Multi-Sensor Composites:** Interactive RGB band mapping and side-by-side comparison slider.
* **Automated PDF Reports:** One-click PDF report generation featuring cryptographic SHA-256 hashes for data provenance.

---

## 🛠️ Architecture Overview

1. **Preprocessing:** Grayscale conversion -> CLAHE contrast balance -> Laplacian edge extraction.
2. **Stage 1 (Coarse Matching):** AKAZE feature extraction -> $k$-NN matching with Lowe's Ratio Test ($0.70$) -> RANSAC homography estimation.
3. **Stage 2 (Fine Optimization):** Sub-pixel affine refinement via NMI maximization using L-BFGS-B.
4. **Validation:** Structural Similarity Index (SSIM), RMSE computation, and spatial distribution entropy analysis.

---

## ⚙️ Quickstart

### Prerequisites
* Python 3.10 or higher

### Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/luna-reg.git](https://github.com/your-username/luna-reg.git)
   cd luna-reg
