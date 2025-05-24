# MBTI Personality Prediction from Social‑Media Posts

Predicting Myers–Briggs Type Indicator (MBTI) personality categories directly from user‑generated text is a lively research area at the intersection of computational psycholinguistics and natural‑language processing. This repository contains reproducible baselines and transformer‑based models for **single‑post and user‑level MBTI prediction**.

---

## ✨ Key Points

* **Granular Challenge** We treat MBTI prediction as **four parallel binary‑classification tasks** (I/E, N/S, T/F, P/J).
* **Rhetoric‑Aware Modeling** Our code includes options to inject rhetorical cues (e.g. sarcasm markers, emoji polarity) into traditional classifiers and into the transformer’s input embeddings.
* **Two Model Tiers**

  1. **Classic ML baselines**: TF‑IDF + SVM, K‑NN, XGBoost
  2. **Contextual transformer**: RoBERTa‑base fine‑tuned with parameter‑efficient adapters
* **Plug‑and‑Play Dataset Loader** works with the popular *MBTI‑5000* dataset (Kaggle) or any CSV/JSONL file in the same format.

---

## 🗂️ Repository Structure

```
.
├── data/                 # (ignored) raw & processed datasets
├── notebooks/
│   ├── MBTI_prediction.ipynb   # RoBERTa training pipeline
│   └── claude_mbti.ipynb       # classical ML baselines & benchmarking
├── src/
│   ├── data_utils.py      # cleaning, sarcasm heuristic, stratified split
│   ├── classic_models.py  # TF‑IDF, SVM, KNN, XGB
│   ├── transformer.py     # RoBERTa fine‑tune / inference
│   └── eval.py            # metrics & plots
├── requirements.txt
└── README.md              # (you are here)
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
$ git clone https://github.com/your‑handle/mbti‑prediction.git
$ cd mbti‑prediction
$ python3 -m venv .venv && source .venv/bin/activate
$ pip install -r requirements.txt
```

> **GPU** ‑ A CUDA‑capable GPU is strongly recommended for the transformer notebook (≈5 GB VRAM is enough).

### 2. Obtain the Dataset

```bash
$ mkdir -p data/raw
$ kaggle datasets download -d dgomonov/new‑mbti‑dataset
$ unzip new‑mbti‑dataset.zip -d data/raw
```

Alternative: drop any CSV with columns `text` and `type` into `data/raw/`.

### 3. Run Baselines

Open `notebooks/claude_mbti.ipynb` and run *all cells*; results are logged to `reports/baselines.md`.

### 4. Fine‑Tune RoBERTa

```bash
$ jupyter lab notebooks/MBTI_prediction.ipynb
```

The notebook walks through:

1. Tokenisation & sarcastic‑cue augmentation
2. Training (≈20 min on a single RTX 3060)
3. Evaluation & visualisation

### 5. Inference on New Text

```python
from src.transformer import MBTIPredictor
mdl = MBTIPredictor("checkpoints/roberta_adapters.pt")
print(mdl("I love organising code and colour‑coding my folders ✨"))
# → ['I', 'N', 'T', 'J']
```

---

## 📊 Results (MBTI‑5000, per‑dimension macro‑F1)

| Model                       | I/E      | N/S      | T/F      | P/J      | Avg      |
| --------------------------- | -------- | -------- | -------- | -------- | -------- |
| TF‑IDF + Linear SVM         | 0.80     | 0.61     | 0.72     | 0.70     | 0.71     |
| KNN (k=9)                   | 0.78     | 0.57     | 0.69     | 0.66     | 0.67     |
| XGBoost                     | 0.82     | 0.63     | 0.74     | 0.72     | 0.73     |
| **RoBERTa‑base + adapters** | **0.86** | **0.69** | **0.78** | **0.76** | **0.77** |

*(Exact numbers may vary ±0.01 due to randomness; see `reports/` for full logs.)*

---

## 🧩 Configuration

Key hyper‑parameters are exposed in the notebooks and `src/config.py`:

* `max_len`: token length (default 128)
* `batch_size`: 32 (classical) / 16 (transformer)
* `lr`: 2e‑5 (transformer)
* `epochs`: 4–6 suffices
* `sarcasm_regex`: pattern list used to tag sarcastic cues

---

## 📑 Citation

If you use this repository, please cite:

```bibtex
@misc{your_handle_2025_mbti,
  title        = {MBTI Personality Prediction from Social Media Posts},
  author       = {Your Name},
  year         = 2025,
  howpublished = {GitHub},
  url          = {https://github.com/your‑handle/mbti‑prediction}
}
```

---

## 📜 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## 🙏 Acknowledgements

* *MBTI‑5000* dataset creators
* HuggingFace 🤗 Transformers
* scikit‑learn team
* OpenAI & Claude community for baseline inspiration

---

Feel free to open an issue or pull request if you encounter any problems or have suggestions!
