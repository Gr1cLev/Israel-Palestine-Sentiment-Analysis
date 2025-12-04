# Israel-Palestine Sentiment Analysis (Reddit 2024-2025)

**Comprehensive sentiment analysis** of Reddit comments about the Israel-Palestine conflict (2024-2025). This research project implements a complete machine learning pipeline from data filtering, preprocessing, labeling, to training classical ML and deep learning models, with **Explainable AI (XAI)** analysis using SHAP for model interpretability.

## 🎯 Project Highlights

- **Temporal Split**: Train on Jan-Oct 2024, Test on Nov-Dec 2024, Inference on 2025 data
- **Multi-Model Approach**: Classical ML (Stacking), Deep Learning (BiLSTM, BERT, Stacking)
- **High Performance**: Best model achieves **98.56% accuracy** and **98.57% macro F1-score**
- **Explainable AI**: SHAP analysis for transparency and interpretability
- **Research-Ready**: Complete documentation, flowchart, and export-ready results

> **Datasets and large artefacts are stored on Google Drive** (too big for the repo). Download raw data from: https://drive.google.com/drive/u/0/folders/1VlPXCwDXPtolbvzqpMy0eLGctCZ11I_o and place the raw CSV in the same folder as the notebooks before running anything. The large folders (`artefacts_2024_window_full/`, `artefacts_2024_window_deep/`, `reports_2024_window_full/`, `xai_outputs/`) are also on Drive.

## 📁 Project Structure & Notebooks

### Main Pipeline Notebooks (Run Sequentially)

| Notebook | Description | Key Outputs |
|----------|-------------|-------------|
| `00_FilterData-Chunk.ipynb` | Filter days 1-10 each month from raw data | `*_window.csv` (2024 & 2025) |
| `01_PreProcessing.ipynb` | Text cleaning, normalization, tokenization, stemming | `*_window_clean.csv` |
| `02_Labeling_TextBlob.ipynb` | Auto-labeling using TextBlob sentiment analyzer | `*_window_labeled_textblob.csv` |
| `03_Training-Tuning-Stacking2024.ipynb` | Classical ML training with hyperparameter tuning | Best models in `artefacts_2024_window_full/` |
| `04_Inference_2025.ipynb` | Predict 2025 data & generate timeseries | `*_window_pred.csv`, timeseries plots |
| `05_DeepLearning_BiLSTM_BERT_Stacking_FIX.ipynb` | Deep learning models (BiLSTM, BERT, Stacking) | Models in `artefacts_2024_window_deep/` |
| `06_ExplainableAI_SHAP_LIME.ipynb` | **XAI analysis using SHAP** for interpretability | `xai_outputs/` folder with analysis |

### Additional Notebooks

- `CV-RingkasanFold-ClassicalML.ipynb` - Summarize cross-validation results to Excel (`reports_2024_window_full/*.xlsx`)
- `flowchart.txt` - Complete project pipeline flowchart in Mermaid format

### Artefacts & Outputs (on Google Drive)

```
artefacts_2024_window_full/          # Classical ML models
├── best_stack_count.joblib           # Best stacking model
├── best_count_lr.joblib              # Best Count+LogisticRegression
├── best_tfidf_svm.joblib             # Best TF-IDF+SVM
├── model_meta.json                   # Model metadata & scores
└── cv_*.csv                          # Cross-validation results

artefacts_2024_window_deep/          # Deep learning models
├── bilstm_bow_model.pt               # BiLSTM model + vocabulary
├── bilstm_results.json               # BiLSTM evaluation metrics
├── bert_model/                       # BERT fine-tuned model
├── bert_tokenizer/                   # BERT tokenizer
├── bert_results.json                 # BERT evaluation metrics
├── stacking_meta_lr.joblib           # DL stacking meta-learner
├── stacking_results.json             # Stacking evaluation metrics
└── stacking_proba_*.npz              # Cached probabilities

xai_outputs/                          # Explainable AI results
├── xai_summary_examples.csv          # Summary of XAI examples
├── xai_token_details_all_examples.csv # Detailed token contributions
├── aggregated_top_tokens.csv         # Top tokens per sentiment class
├── error_analysis_detailed.csv       # Misclassification analysis
├── model_comparison_ml_vs_bert.csv   # Model comparison (optional)
├── aggregated_tokens_bar_plot.png    # Visualization
└── shap_ml_feature_importance_spaced.png # SHAP global importance

reports_2024_window_full/             # Cross-validation summaries
├── hyperparameter_tuning_from_single_csv.xlsx
└── cross_validation_for_slide.xlsx
```

## 🔄 Complete Data Flow Pipeline

For detailed flowchart, see `flowchart.txt` (Mermaid format).

### 1. **Data Collection & Filtering (Notebook 00)**
- **Input**: `reddit_opinion_PSE_ISR.csv` (raw 2024+2025 combined)
- **Process**: 
  - Stream processing with `chunksize=1,000,000`
  - Filter days 1-10 each month (window approach)
  - Parse datetime, light deduplication on `comment_id`
  - Split by year (2024 vs 2025)
- **Output**: `reddit_opinion_PSE_ISR_2024_window.csv`, `reddit_opinion_PSE_ISR_2025_window.csv`
- **Key Columns**: `comment_id`, `created_time`, `self_text`, `score`, `subreddit`

### 2. **Text Preprocessing (Notebook 01)**
- **Dependencies**: `ftfy`, `ekphrasis`, `emoji`, `nltk`, `scikit-learn`
- **Pipeline Steps**:
  1. **Fix encoding** (ftfy)
  2. **Social media normalization** (ekphrasis: URLs, HTML, mentions, hashtags)
  3. **Emoji replacement** (emoji library + custom mapping)
  4. **Slang normalization** (custom lexicon with fallback)
  5. **Tokenization** (ToktokTokenizer)
  6. **Stopword removal** (NLTK English stopwords)
  7. **Stemming** (PorterStemmer)
  8. **Join to final_text**
- **Output**: `*_window_clean.csv` with `final_text` column

### 3. **Auto-Labeling (Notebook 02)**
- **Tool**: TextBlob PatternAnalyzer
- **Thresholds**:
  - Polarity ≥ +0.05 → **Positif**
  - Polarity ≤ -0.05 → **Negatif**
  - Otherwise → **Netral**
- **Output**: `reddit_opinion_PSE_ISR_2024_window_labeled_textblob.csv` and `*_2025_*.csv`
- **Quality Check**: Label distribution and polarity histogram shown in-notebook

### 4. **Classical ML Training (Notebook 03)**
- **Temporal Split**: 
  - **Train**: Jan-Oct 2024 (months 2024-01 to 2024-10)
  - **Test**: Nov-Dec 2024 (months 2024-11, 2024-12)
- **Feature Extraction**: CountVectorizer, TF-IDF
- **Models Tested**:
  - ComplementNB (Naive Bayes)
  - LinearSVC (Support Vector Machine)
  - LogisticRegression
  - **StackingClassifier** (NB + Calibrated SVM + LR, meta=LR)
- **Hyperparameter Tuning**: 
  - `RandomizedSearchCV` with 5-fold CV
  - Metric: Macro F1-score
  - Results saved to `cv_*.csv` and `cv_results_2024_window.json`
- **Best Model**: `stack_count` (Stacking with CountVectorizer)
  - **Test Accuracy**: 95.28%
  - **Test Macro F1**: 95.25%
  - **Best CV F1**: 94.86%
- **Output**: Models saved to `artefacts_2024_window_full/best_*.joblib`

### 5. **2025 Inference (Notebook 04)**
- **Input**: `reddit_opinion_PSE_ISR_2025_window_clean.csv`
- **Process**:
  - Load best pipeline from `model_meta.json`
  - Predict sentiment for all 2025 comments
  - Generate monthly aggregations
- **Output**:
  - `reddit_opinion_PSE_ISR_2025_window_pred.csv` (predictions)
  - `sentiment_timeseries_2025_count.csv` (monthly counts)
  - `sentiment_timeseries_2025_share.csv` (monthly percentages)
  - Timeseries visualization plots

### 6. **Deep Learning Models (Notebook 05)** 🚀 **GPU Required**
- **Same Temporal Split**: Jan-Oct train, Nov-Dec test

#### 6.1 BiLSTM Model
- **Architecture**:
  - Feature extraction: CountVectorizer (BOW vocabulary)
  - Embedding dimension: 128
  - Hidden dimension: 256 (bidirectional)
  - Dropout: 0.3
  - Max sequence length: 128
- **Training**: 5 epochs, batch size 64, LR 2e-3
- **Results**: 
  - **Test Accuracy**: 97.97%
  - **Test Macro F1**: 97.98%
- **Saved**: `bilstm_bow_model.pt`, `bilstm_results.json`

#### 6.2 BERT Fine-tuning
- **Model**: `bert-base-uncased`
- **Configuration**:
  - Max length: 128
  - Batch size: 16
  - Epochs: 3
  - Learning rate: 2e-5
- **Results**: 
  - **Test Accuracy**: 98.56%
  - **Test Macro F1**: 98.57%
- **Saved**: `bert_model/`, `bert_tokenizer/`, `bert_results.json`

#### 6.3 Deep Learning Stacking
- **Approach**: BiLSTM + BERT probabilities → Logistic Regression meta-learner
- **Results**: 
  - **Test Accuracy**: 98.41%
  - **Test Macro F1**: 98.42%
- **Saved**: `stacking_meta_lr.joblib`, `stacking_results.json`, probability caches

### 7. **Explainable AI Analysis (Notebook 06)** 🔍 **NEW**
- **Method**: SHAP (SHapley Additive exPlanations)
- **Models Analyzed**:
  - Traditional ML (Best Stacking model)
  - Deep Learning (BERT) - optional
  
#### 7.1 Sample Selection Strategy
- 1 example: Very Positive (high confidence)
- 1 example: Very Negative (high confidence)
- 1 example: Neutral
- 1-2 examples: Misclassified (for error analysis)

#### 7.2 XAI Outputs
- **Token-level explanations**: SHAP values showing contribution of each word
- **Visualization**: 
  - SHAP text plots (color-coded word importance)
  - SHAP bar plots (top features per sentiment class)
  - Aggregated token importance charts
- **Tables**:
  - Summary table with dominant tokens per example
  - Detailed token contribution tables
  - Error analysis with insights (negation, mixed sentiment, sarcasm detection)
  - Aggregated top tokens per sentiment class
- **Model Comparison** (if BERT available):
  - Side-by-side comparison: Traditional ML vs BERT
  - Insight: ML focuses on individual words, BERT captures context

#### 7.3 Research Value
- **Transparency**: Understand why model makes specific predictions
- **Error Analysis**: Identify patterns in misclassifications
- **Feature Validation**: Verify important tokens align with domain knowledge
- **Stakeholder Communication**: Explain model decisions to non-technical audiences

### 8. **Cross-Validation Summary (CV-RingkasanFold)**
- Reads `cv_*.csv` files
- Generates Excel reports: `hyperparameter_tuning_from_single_csv.xlsx`, `cross_validation_for_slide.xlsx`
- Saved to `reports_2024_window_full/`

## 📊 Model Performance Summary

| Model | Test Accuracy | Test Macro F1 | Notes |
|-------|--------------|---------------|-------|
| **Classical ML** |
| Count + ComplementNB | ~90% | ~89% | Baseline |
| Count + LinearSVC | ~93% | ~92% | Good performance |
| Count + LogisticRegression | ~94% | ~93% | Strong baseline |
| TF-IDF + Models | Similar | Similar | Comparable to Count |
| **Stacking (Count)** ⭐ | **95.28%** | **95.25%** | **Best Classical** |
| **Deep Learning** |
| BiLSTM (BOW vocab) | 97.97% | 97.98% | Strong DL baseline |
| **BERT fine-tuned** 🏆 | **98.56%** | **98.57%** | **Best Overall** |
| DL Stacking (BiLSTM+BERT) | 98.41% | 98.42% | Ensemble approach |

**Key Insight**: BERT achieves the highest performance, showing the power of pre-trained transformers for sentiment analysis. However, the classical ML stacking model offers a good balance of performance and computational efficiency.

## 🧠 Explainable AI Insights

From SHAP analysis (`06_ExplainableAI_SHAP_LIME.ipynb`):

### Top Tokens per Sentiment Class

**Positif**: support, peace, hope, freedom, right, protect, help, agree, good, thank
**Netral**: think, people, question, debate, discuss, consider, situation, complex, understand
**Negatif**: attack, kill, war, genocide, terrorist, violence, hate, destroy, blame, wrong

### Error Analysis Patterns
Common misclassification causes:
- **Negation**: "not good", "isn't acceptable" (sarcasm/negation not captured)
- **Mixed Sentiment**: Comments containing both positive and negative words
- **Context Complexity**: Long texts with multiple sentiments
- **Sarcasm**: Requires deeper contextual understanding

### Model Comparison
- **Traditional ML**: Focuses on individual high-weight words (tf-idf/count based)
- **BERT**: Captures contextual relationships and semantic meaning
- **Overlap**: Words important in both models are strong sentiment indicators
## 🚀 Quickstart Guide

### 1. Environment Setup
**Python 3.10+ recommended**

```bash
# Core dependencies
pip install pandas numpy matplotlib tqdm scikit-learn imbalanced-learn joblib

# Text processing
pip install ftfy ekphrasis emoji textblob nltk

# Deep learning (optional, for Notebook 05)
pip install torch torchvision torchaudio transformers sentencepiece

# Explainable AI (for Notebook 06)
pip install shap
```

**NLTK Data Download** (run once):
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**GPU Setup** (for Deep Learning):
- NVIDIA GPU with CUDA support required for Notebook 05
- Install PyTorch with CUDA: https://pytorch.org/get-started/locally/

### 2. Download Data
1. Download raw data from Google Drive: [Link](https://drive.google.com/drive/u/0/folders/1VlPXCwDXPtolbvzqpMy0eLGctCZ11I_o)
2. Place `reddit_opinion_PSE_ISR.csv` in the **same folder as notebooks**
3. (Optional) Download pre-trained artefacts to skip training

### 3. Run Notebooks Sequentially

#### **Essential Pipeline** (for basic analysis):
```
00 → 01 → 02 → 03 → 04
```
- `00_FilterData-Chunk.ipynb` - Filter window data
- `01_PreProcessing.ipynb` - Clean text
- `02_Labeling_TextBlob.ipynb` - Auto-label sentiment
- `03_Training-Tuning-Stacking2024.ipynb` - Train classical ML
- `04_Inference_2025.ipynb` - Predict 2025 data

#### **Extended Pipeline** (for complete research):
```
00 → 01 → 02 → 03 → 04 → 05 → 06
```
- `05_DeepLearning_BiLSTM_BERT_Stacking_FIX.ipynb` - Deep learning models (GPU required)
- `06_ExplainableAI_SHAP_LIME.ipynb` - XAI analysis with SHAP

#### **Additional Analysis**:
- `CV-RingkasanFold-ClassicalML.ipynb` - CV summary to Excel

### 4. Quick Inference (Skip Training)
If you have pre-trained models from Google Drive:

```python
# Load best classical model
import joblib
pipeline = joblib.load('artefacts_2024_window_full/best_stack_count.joblib')

# Predict
texts = ["Your text here"]
predictions = pipeline.predict(texts)
probabilities = pipeline.predict_proba(texts)
```

### 5. Configuration Notes
- **Chunking**: Default `chunksize=1_000_000` for memory efficiency (adjust if needed)
- **Parallelism**: Set `n_jobs=1` or configure `OMP_NUM_THREADS` to control CPU usage
- **Paths**: All notebooks assume files are in the same directory
- **GPU**: Notebook 05 requires CUDA; will assert if GPU not available

## 📈 Expected Runtime

| Notebook | Time (CPU) | Time (GPU) | Memory |
|----------|-----------|-----------|--------|
| 00 - Filter | ~5-10 min | N/A | ~4 GB |
| 01 - Preprocess | ~10-15 min | N/A | ~4 GB |
| 02 - Labeling | ~5 min | N/A | ~4 GB |
| 03 - Training | ~30-60 min | N/A | ~8 GB |
| 04 - Inference | ~5 min | N/A | ~4 GB |
| 05 - Deep Learning | N/A | ~2-3 hours | ~8-12 GB VRAM |
| 06 - XAI | ~20-30 min | ~10-15 min | ~8 GB |

*Times are approximate and depend on hardware*

## 📝 Important Notes & Best Practices

### Data Requirements
- **Required columns**: `comment_id`, `created_time`, `self_text`, `score`, `subreddit`
- **Text column**: Preprocess will use `final_text` or `self_text` (auto-detected)
- **Missing data**: Empty or URL-only rows are automatically dropped
- **Encoding**: UTF-8 encoding handled by ftfy

### Model Persistence
- All outputs/artefacts are saved in the same directory as notebooks
- **Don't move files** while notebooks are running to prevent path errors
- Models are saved with `.joblib` (classical ML) or `.pt` (PyTorch) format
- Metadata in `.json` files for easy loading

### Performance Optimization
- **Memory**: Use chunking for large files (`chunksize=1_000_000`)
- **CPU**: Limit threads via `n_jobs=1` or environment variables:
  ```python
  import os
  os.environ['OMP_NUM_THREADS'] = '4'
  os.environ['MKL_NUM_THREADS'] = '4'
  ```
- **GPU**: Deep learning (Notebook 05) requires NVIDIA GPU with CUDA
- **Caching**: XAI analysis caches SHAP values to avoid recomputation

### Lexicons & Resources
- **Slang lexicon**: Falls back to built-in mini lexicon if external file missing
- **Emoji mapping**: Built into pipeline with fallback support
- **Stopwords**: NLTK English stopwords (download required)
- **Stemmer**: Porter Stemmer from NLTK

### Reproducibility
- **Random seed**: Set to 42 across all notebooks
- **Temporal split**: Fixed train/test split by month (no data leakage)
- **Version control**: All hyperparameters logged in artefacts
- **Flowchart**: Complete pipeline documented in `flowchart.txt`

## 🔬 Research Applications

This project demonstrates:

1. **Complete ML Pipeline**: From raw data to deployment-ready models
2. **Temporal Validation**: Proper time-series split to avoid data leakage
3. **Multi-Model Comparison**: Classical ML vs Deep Learning
4. **Explainable AI**: SHAP analysis for model interpretability
5. **Production-Ready**: Modular notebooks with reusable components
6. **Documentation**: Comprehensive README, flowchart, and inline comments

### Suitable For:
- **Academic Research**: Sentiment analysis methodology papers
- **Thesis/Dissertation**: Complete pipeline with XAI analysis
- **Industry Projects**: Production-ready sentiment analysis system
- **Education**: Learning end-to-end ML project structure
- **Social Media Analytics**: Reddit/Twitter sentiment monitoring

## 📚 Key References & Technologies

### Libraries & Frameworks
- **Text Processing**: ftfy, ekphrasis, emoji, NLTK
- **ML**: scikit-learn, imbalanced-learn
- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **XAI**: SHAP (SHapley Additive exPlanations)
- **Visualization**: matplotlib, seaborn

### Models Used
- **Classical ML**: ComplementNB, LinearSVC, LogisticRegression, StackingClassifier
- **Deep Learning**: BiLSTM (custom), BERT-base-uncased (fine-tuned)
- **Ensemble**: Stacking meta-learner (LogisticRegression)

### Methodologies
- **Feature Extraction**: CountVectorizer, TF-IDF
- **Hyperparameter Tuning**: RandomizedSearchCV with 5-fold CV
- **Evaluation**: Accuracy, Macro F1-score, Confusion Matrix
- **Interpretability**: SHAP values, token importance analysis

## 🤝 Contributing & Support

For questions, issues, or contributions:
1. Check existing documentation in notebooks
2. Review `flowchart.txt` for pipeline overview
3. Examine `model_meta.json` for model details
4. Refer to XAI outputs in `xai_outputs/` for interpretability

## 📄 License & Citation

If you use this project in your research, please cite appropriately and ensure proper attribution for:
- Original Reddit data source
- Libraries and frameworks used
- SHAP methodology for XAI analysis

---

**Project Status**: ✅ Complete | **Last Updated**: December 2025

**Pipeline Stages**: 8 notebooks | **Best Model**: BERT (98.56% accuracy) | **XAI**: SHAP analysis included
