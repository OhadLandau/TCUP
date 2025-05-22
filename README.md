# TCUP – Tissue‑of‑Origin Prediction

**TCUP** (Tissue Classification Using Probabilities) is an open‑access Dash web‑application and model suite that predicts the most likely tissue of origin for:

* cancers of unknown primary (CUP)  
* any transcriptomic sample (e.g. TCGA, GTEx, metastatic, healthy)

The method couples a **Contrastive Auto‑Encoder (CAE)** for representation learning with a **Siamese Neural Network (SNN)** meta‑learner trained on curated TCGA, GTEx and metastatic cohorts.  
For full methodology see the pre‑print:

> Landau O., Rubin E. “TCUP – Open Access Tool to Predict Tissue of Origin for Cancer of Unknown Primary (CUP) and Unknown Transcriptomic Samples” (in preparation, 2025).

---

## 🌳 Repository layout

```
.
├─ app/                     # Dash front‑end
│   ├─ app.py
│   └─ assets/              # CSS, logo, favicon
├─ models/                  # Pre‑trained weights & helpers
│   ├─ tcup_cancer.pkl
│   ├─ tcup_healthy.pkl
│   ├─ medians_cancer.csv
│   ├─ medians_healthy.csv
│   └─ label_encoder.pkl
├─ Raw Model Script/        # Original end‑to‑end training pipeline
│   └─ TCUP_raw_training_script.py
├─ images/                  # Screenshots for this README
│   ├─ landing_page.png
│   └─ results_page.png
├─ requirements.txt
└─ README.md                # <–– you are here
```

### Key artefacts

| Path | Purpose |
|------|---------|
| **models/tcup_cancer.pkl** | CAE + SNN stack trained on *cancer* samples |
| **models/tcup_healthy.pkl** | Equivalent model trained on *healthy* tissues |
| **medians_cancer.csv** / **medians_healthy.csv** | Per‑gene median expression used for graceful imputation when a gene is missing from the user upload |
| **Raw Model Script/** | Reproducible pipeline to retrain TCUP from scratch (data download ➜ preprocessing ➜ training ➜ evaluation) |

---

## 🚀 Quick start

```bash
# 1. clone repo
git clone https://github.com/<your‑user>/tcup.git
cd tcup

# 2. install deps
python -m venv venv && source venv/bin/activate  # optional
pip install -r requirements.txt

# 3. launch Dash server
python app/app.py
```

The app starts on **http://127.0.0.1:8050** by default.

---

## 🖼  UI tour

| Landing | Results |
|---------|---------|
| ![Landing page](images/landing_page.png) | ![Results page](images/results_page.png) |

* **Landing page** – drag‑and‑drop a CSV/TSV where the **first column is `sample_id`** and the rest are *HGNC gene symbols*. Choose **Cancer** vs **Healthy** to route the input through the appropriate model.  
* **Results page** – shows:  
  1. *Top‑N* class probabilities (barplot)  
  2. Predicted tissue + model‑confidence (bold)  
  3. **TCUP accuracy** – historical accuracy for that tissue on an unseen test set (≈ macro F1 for the given label).  
  4. The 20 most influential genes in the decision (blue ▲ = over‑expressed; red ▼ = under‑expressed).

---

## 📈 Interpreting the output

* **Predicted tissue** – the label (e.g. `SKCM_TCGA`) with the highest posterior probability.  
* **Probability** – the softmax score from the SNN head (0‑1).  
* **TCUP accuracy** – *held‑out* test‑set accuracy **for that tissue** only.  
  *Example*: “TCUP accuracy ≈ 96.7 %” means that, across 2 000 unseen melanoma (SKCM) samples, 96.7 % were correctly classified by TCUP.

If your sample’s probability is low (<0.4) or multiple tissues cluster tightly, treat the prediction with caution and consider further histopathological or molecular work‑up.

---

## 🛠  Retraining

1. Download raw expression matrices (TCGA, GTEx, metastatic) – details inside *Raw Model Script* .  
2. Edit `config.yml` for paths / hyper‑parameters.  
3. Run `python TCUP_raw_training_script.py`.  
   The pipeline is fully resumable and will write:
   * new model binaries to `models/`
   * metrics & plots to `reports/`

---

## 📂 Placing the screenshots

Put the two PNGs **inside `images/`** at repo root:

```
images/
├─ landing_page.png
└─ results_page.png
```

Because the README uses the relative Markdown links `images/landing_page.png` and `images/results_page.png`, GitHub will automatically render them once pushed.

---

## 📜 License & citation

TCUP is released under the **MIT License** – see [LICENSE](LICENSE).

If you use TCUP in academic work, please cite the forthcoming pre‑print:

```
Landau O., Rubin E. TCUP – Open Access Tool to Predict Tissue of Origin
for Cancer of Unknown Primary (CUP) and Unknown Transcriptomic Samples. 2025.
```

---

*Made with ❤️  by Ohad Landau & Eitan Rubin*
