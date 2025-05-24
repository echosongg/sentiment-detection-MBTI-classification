# MBTI Personality Prediction from Social Media Posts

Predicting Myers–Briggs Type Indicator (MBTI) personality categories directly from user-generated text is a challenging task at the intersection of computational psychology and natural language processing. This repository contains reproducible implementations of multiple deep learning models for **MBTI personality prediction**.

---

## ✨ Key Features

* **Multi-Task Learning** We treat MBTI prediction as **four parallel binary classification tasks** (E/I, S/N, T/F, J/P).
* **Comprehensive Model Comparison** Implementation of both classical baselines and state-of-the-art transformer models.
* **Two Main Components**
  1. **Benchmark Models**: TF-IDF + SVM, CNN, BiLSTM + GloVe, BERT-base, Pure RoBERTa
  2. **Proposed Model**: RCNN-RoBERTa (combining RoBERTa with bidirectional LSTM in an RCNN architecture)
* **Robust Evaluation** Comprehensive metrics including accuracy, precision, recall, and F1-score for both overall performance and individual MBTI dimensions.

---
---

## 🚀 Quick Start

### 1. Clone & Install

```bash
$ git clone https://github.com/your-username/mbti-prediction.git
$ cd mbti-prediction
$ python3 -m venv .venv && source .venv/bin/activate  # Linux/Mac
$ python -m venv .venv && .venv\Scripts\activate     # Windows
$ pip install -r requirements.txt
```

> **GPU Recommendation** - A CUDA-capable GPU is strongly recommended for transformer models (≈4-8 GB VRAM).

### 2. Download the Dataset

Visit the [MBTI Dataset on Kaggle](https://www.kaggle.com/datasets/datasnaek/mbti-type) and download the dataset.

```bash
$ mkdir -p data
# Download mbti_1.csv from Kaggle and place it in the data/ folder
```

**Dataset Format**: The CSV should contain columns:
- `type`: MBTI personality type (e.g., "ENFP", "ISTJ")
- `posts`: User's social media posts (text)

### 3. Run Benchmark Models

```bash
$ python benchmark_models.ipynb
```

This will run all baseline models:
- TF-IDF + SVM
- CNN with embeddings
- BERT-base
- Pure RoBERTa
- KNN+XGboost

### 4. Run Proposed RCNN-RoBERTa Model

```bash
$ python rcnn_roberta_model.ipynb
```

This runs our proposed model that combines:
- RoBERTa transformer for contextual embeddings
- Bidirectional LSTM for sequence modeling
- Max pooling and fully connected layers for classification

### 5. Model Training Details

Each model will:
1. **Preprocess** text (lowercase, remove URLs, clean whitespace)
2. **Split** data into train/validation/test (70/15/15)
3. **Train** for 5 epochs with early stopping
4. **Evaluate** on test set with comprehensive metrics
5. **Save** best model weights and training plots

---

## 📊 Expected Results

Based on MBTI dataset performance (results may vary ±0.02 due to randomness):

| Model                    | Overall F1 | E/I F1 | S/N F1 | T/F F1 | J/P F1 |
|--------------------------|------------|--------|--------|--------|--------|
| TF-IDF + SVM            | ~0.65      | ~0.78  | ~0.55  | ~0.68  | ~0.62  |
| CNN                     | ~0.68      | ~0.80  | ~0.58  | ~0.70  | ~0.65  |
| BiLSTM + GloVe          | ~0.70      | ~0.82  | ~0.60  | ~0.72  | ~0.67  |
| BERT-base               | ~0.74      | ~0.84  | ~0.65  | ~0.76  | ~0.71  |
| Pure RoBERTa            | ~0.76      | ~0.85  | ~0.67  | ~0.78  | ~0.73  |
| **RCNN-RoBERTa (Ours)** | **~0.78**  | **~0.87** | **~0.69** | **~0.80** | **~0.75** |

*Note: I/E dimension typically shows highest performance, while S/N is most challenging across all models.*

---

## 🧩 Model Architectures

### Benchmark Models
- **TF-IDF + SVM**: Classical baseline with term frequency features
- **CNN**: Convolutional neural network with word embeddings
- **BiLSTM**: Bidirectional LSTM with GloVe embeddings
- **BERT-base**: Fine-tuned BERT transformer
- **RoBERTa**: Fine-tuned RoBERTa transformer

### Proposed RCNN-RoBERTa
Our novel architecture combines:
1. **RoBERTa** for rich contextual representations
2. **Bidirectional LSTM** for sequential modeling
3. **Feature Concatenation** of RoBERTa and LSTM outputs
4. **Max Pooling** for sequence-level representation
5. **Multi-task Classification** heads for four MBTI dimensions

---

## ⚙️ Configuration

Key hyperparameters (modify in respective Python files):

```python
# Common settings
BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 5
RANDOM_SEED = 42

# Transformer models
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-5

# Classical models
VOCAB_SIZE = 10000  # CNN, BiLSTM
TFIDF_FEATURES = 5000  # SVM
```

---

## 📁 Output Files

After training, each model generates:
- `best_[model_name]_model.pth`: Best model weights
- `[model_name]_training_loss.png`: Training/validation loss curves
- Console output with detailed evaluation metrics

---

## 🔧 Requirements

```txt
torch>=1.9.0
transformers>=4.20.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.60.0
```

---

## 📖 Usage Example

```python
# Load and preprocess data
data = load_data('data/mbti_1.csv')

# Split into train/validation/test
X_train, X_val, X_test, y_train, y_val, y_test = split_data(data)

# Initialize model
model = RCNNRoBERTa(num_classes=4)

# Train
train_model(model, train_dataloader, val_dataloader, optimizer, criterion, epochs=5)

# Evaluate
results = evaluate_model(model, test_dataloader)
print(f"Overall F1: {results['overall']['f1']:.4f}")
```

---

## 🎯 Research Contribution

This work contributes:
1. **Comprehensive Benchmark**: Implementation of 6 different approaches for MBTI prediction
2. **Novel Architecture**: RCNN-RoBERTa combining transformer and recurrent networks
3. **Thorough Evaluation**: Per-dimension analysis revealing model strengths/weaknesses
4. **Reproducible Code**: Clean, documented implementations for research reproducibility

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements

* [MBTI Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) creators
* Hugging Face 🤗 Transformers library
* PyTorch team
* scikit-learn contributors

---

## 📧 Contact

Feel free to open an issue if you encounter any problems or have suggestions for improvements!

---

**Note**: Update the file paths in the Python scripts to match your local data directory before running.
