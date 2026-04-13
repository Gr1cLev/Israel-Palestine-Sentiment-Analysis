# Sentiment Analysis Reddit Comments on Palestine Israel Conflict Using Machine Learning and Deep Learning

## 1. Title

**Project Name:** Sentiment Analysis Reddit Comments on Palestine Israel Conflict Using Machine Learning and Deep Learning  
**Data Source:** Reddit comments (via Kaggle)  
**Topic:** Sentiment Analysis, NLP, Deep Learning, Explainable AI

---

## 2. Code / Dataset Description

This project implements a complete machine learning pipeline for sentiment analysis of Reddit comments related to the Israel–Palestine conflict (2024–2025). It covers data filtering, text preprocessing, auto-labeling, classical ML and deep learning model training, 2025 inference/trend summarization, and Explainable AI (XAI) analysis using SHAP.

> **Note:** Large datasets and model artefacts are stored on Google Drive (too large for GitHub).  
> Download from: [Google Drive](https://drive.google.com/drive/u/0/folders/1VlPXCwDXPtolbvzqpMy0eLGctCZ11I_o)

---

## 3. Dataset Information

- **Raw Reddit Dataset:**  
  Reddit comments on the Israel–Palestine conflict, sourced from Kaggle (daily updated).  
  [Kaggle Dataset](https://www.kaggle.com/datasets/asaniczka/reddit-on-israel-palestine-daily-updated)

- **Key Columns:** `comment_id`, `created_time`, `self_text`, `score`, `subreddit`

- **Split Strategy:**
  - Train: January–October 2024
  - Test: November–December 2024
  - Inference: 2025 data

---

## 4. Code Description

| File Name | Description |
|---|---|
| `00_FilterData-Chunk.ipynb` | Filter days 1–10 each month from raw data (window approach) |
| `01_PreProcessing.ipynb` | Text cleaning, normalization, tokenization, stemming |
| `02_Labeling_TextBlob.ipynb` | Auto-labeling using TextBlob PatternAnalyzer |
| `03_Training-Tuning-Stacking2024.ipynb` | Classical ML training with hyperparameter tuning (RandomizedSearchCV) |
| `04_Inference_2025.ipynb` | Predict 2025 data & generate sentiment timeseries |
| `05_DeepLearning_BiLSTM_BERT_Stacking_FIX.ipynb` | Deep learning models: BiLSTM, BERT fine-tuning, DL Stacking |
| `06_ExplainableAI_SHAP_LIME.ipynb` | XAI analysis using SHAP for model interpretability |
| `CV-RingkasanFold-ClassicalML.ipynb` | Summarize cross-validation results to Excel |

---

## 5. Usage Instructions

1. **Download Dataset:**  
   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/asaniczka/reddit-on-israel-palestine-daily-updated) and place `reddit_opinion_PSE_ISR.csv` in the same folder as the notebooks.

2. **Install Dependencies:**  
   See Section 6 for required packages.

3. **Run Essential Pipeline:**  
   `00 → 01 → 02 → 03 → 04`

4. **Run Full Pipeline (with Deep Learning & XAI):**  
   `00 → 01 → 02 → 03 → 04 → 05 → 06`  
   ⚠️ Notebook 05 requires an NVIDIA GPU with CUDA support.

5. **Cross-Validation Summary:**  
   Run `CV-RingkasanFold-ClassicalML.ipynb` after Notebook 03.

---

## 6. Requirements

```bash
pip install pandas numpy matplotlib tqdm scikit-learn imbalanced-learn joblib
pip install ftfy ekphrasis emoji textblob nltk
pip install torch torchvision torchaudio transformers sentencepiece
pip install shap
```

**NLTK Data (run once):**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**Key Libraries:**
- `pandas`, `numpy` — data handling
- `scikit-learn`, `imbalanced-learn` — classical ML
- `torch`, `transformers` — deep learning (BiLSTM, BERT)
- `shap` — Explainable AI
- `ftfy`, `ekphrasis`, `emoji`, `nltk` — text preprocessing

---

## 7. Methodology for Code Usage

1. Use `00_FilterData-Chunk.ipynb` to filter windowed data (days 1–10/month).
2. Use `01_PreProcessing.ipynb` to clean and normalize text.
3. Use `02_Labeling_TextBlob.ipynb` to auto-label sentiment (Positive/Neutral/Negative).
4. Use `03_Training-Tuning-Stacking2024.ipynb` to train and tune classical ML models.
5. Use `04_Inference_2025.ipynb` to predict 2025 data and generate timeseries.
6. Use `05_DeepLearning_BiLSTM_BERT_Stacking_FIX.ipynb` for deep learning (GPU required).
7. Use `06_ExplainableAI_SHAP_LIME.ipynb` for SHAP-based interpretability analysis.

---

## 8. Citation

Not applicable.

---

## 9. License & Contribution Guidelines

This project is licensed under the **MIT License**.  
See the [LICENSE](./LICENSE) file for details.

---

## 10. Code Repository or DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19535956.svg)](https://doi.org/10.5281/zenodo.19535956)
