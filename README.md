# Israel-Palestine Sentiment Analysis (Reddit 2024-2025)

Sentiment analysis of Reddit comments about the Israel-Palestine conflict (2024-2025). The project filters comments (days 1-10 each month), cleans text, auto-labels with TextBlob, trains classical models (Count/TF-IDF + NB/SVM/LR + stacking), tests deep learning (BiLSTM, BERT, stacking), then runs inference and monthly trends for 2025.

> **Datasets and large artefacts are stored on Google Drive** (too big for the repo). Download raw data from: https://drive.google.com/drive/u/0/folders/1VlPXCwDXPtolbvzqpMy0eLGctCZ11I_o and place the raw CSV in the same folder as the notebooks before running anything. The large folders (`artefacts_2024_window_full/`, `artefacts_2024_window_deep/`, `reports_2024_window_full/`) are also on Drive.

## Structure & Key Artefacts
- `00_FilterData-Chunk.ipynb` - filter days 1-10 each month from the raw file.
- `01_PreProcessing.ipynb` - cleaning + normalization + stopwords + stemming, outputs `..._window_clean.csv`.
- `02_Labeling_TextBlob.ipynb` - auto-labeling (TextBlob) -> `..._window_labeled_textblob.csv`.
- `03_Training-Tuning-Stacking2024.ipynb` - classical models, CV/tuning, save best pipeline.
- `04_Inference_2025.ipynb` - load best pipeline, predict 2025, timeseries summary.
- `05_DeepLearning_BiLSTM_BERT_Stacking_FIX.ipynb` - BiLSTM, BERT, and stacking ensemble (GPU needed).
- `CV-RingkasanFold-ClassicalML.ipynb` - summarize CV to Excel (`reports_2024_window_full/*.xlsx`).
- Model artefacts (on Google Drive):
  - `artefacts_2024_window_full/` holds the best classical pipeline `sentiment_pipeline_2024win_stack_count.pkl` + `model_meta.json`.
  - `artefacts_2024_window_deep/` holds BiLSTM/BERT/stacking models and metrics.
  - `reports_2024_window_full/` holds CV Excel summaries.

## Data Flow
All outputs are written alongside the notebooks.

1. **Raw**: place `reddit_opinion_PSE_ISR.csv` (2024+2025 combined) at repo root.
2. **Window filter (00)**  
   - Reads `comment_id, created_time, self_text, score, subreddit`.  
   - Streams `chunksize=1_000_000`, parses datetime, keeps days 1-10, splits 2024 vs 2025, light per-year dedup on `comment_id`.  
   - Outputs: `reddit_opinion_PSE_ISR_2024_window.csv`, `reddit_opinion_PSE_ISR_2025_window.csv`.
3. **Preprocess (01)**  
   - Deps: `ftfy, ekphrasis, emoji, scikit-learn, nltk (stemmer/tokenizer)`.  
   - Pipeline: clean URL/HTML/mention/non-print + lowercase + social normalize (ekphrasis + emoji map) + slang replace (optional lexicon, fallback provided) + tokenize + stopword removal + Porter stem + join as `final_text`; fill `month` from `created_time`.  
   - Outputs: `..._window_clean.csv` (2024 & 2025).
4. **Labeling (02)**  
   - Dep: `textblob`.  
   - Polarity thresholds: `>= +0.05 = Positif`, `<= -0.05 = Negatif`, else Netral.  
   - Outputs: `reddit_opinion_PSE_ISR_2024_window_labeled_textblob.csv` and `reddit_opinion_PSE_ISR_2025_window_labeled_textblob.csv` (with label distribution/polarity previews in-notebook).
5. **Classical training (03)**  
   - Train: Jan-Oct 2024, Test: Nov-Dec 2024 (`month` column).  
   - Candidates: Count/TF-IDF + ComplementNB, LinearSVC, Logistic Regression; Stacking (NB + Calibrated SVM + LR, meta=LR).  
   - Hyperparams: `RandomizedSearchCV` (F1-macro, `n_jobs=1`), CV results saved to `artefacts_2024_window_full/cv_*.csv` and `cv_results_2024_window.json`.  
   - **Winner**: `stack_count` (Count 1-gram, min_df=5, max_df=0.9) with Calibrated SVM (C=5, max_iter=10000, tol=0.003), NB (alpha=0.5), LR (C=5), meta LR (C=1).  
   - Key scores (hold-out Nov-Dec): Accuracy 0.9528, Macro F1 0.9525 (`artefacts_2024_window_full/model_meta.json`). Best CV: 0.9486.  
   - Pipeline saved to `sentiment_pipeline_2024win_stack_count.pkl` + meta JSON (label order, text column, etc.).
6. **Inference 2025 (04)**  
   - Picks best pipeline via `model_meta.json`/`cv_results_2024_window.json`, loads `.joblib/.pkl`.  
   - Input: `reddit_opinion_PSE_ISR_2025_window_clean.csv`.  
   - Output: predictions -> `reddit_opinion_PSE_ISR_2025_window_pred.csv`; monthly counts (`sentiment_timeseries_2025_count.csv`) & shares (`..._share.csv`); timeseries plot shown inline (PNG optional).
7. **Deep learning (05)** - GPU required  
   - Same split (Jan-Oct train, Nov-Dec test).  
   - **BiLSTM** (CountVectorizer vocab -> seq, max_len=128, embed_dim=128, hidden=256, dropout=0.3, epochs=5).  
     - Results: Accuracy 0.9797, F1 0.9798 (`bilstm_results.json`), model `bilstm_bow_model.pt`.  
   - **BERT base uncased** (max_len=128, bs=16, epochs=3, lr=2e-5).  
     - Results: Accuracy 0.9856, F1 0.9857 (`bert_results.json`), artefacts `bert_model/`, `bert_tokenizer/`.  
   - **Stacking** (BiLSTM + BERT probs -> Logistic Regression).  
     - Results: Accuracy 0.9841, F1 0.9842 (`stacking_results.json`), meta `stacking_meta_lr.joblib`, proba cache `stacking_proba_*.npz`.
8. **CV recap**  
   - `CV-RingkasanFold-ClassicalML.ipynb` reads `cv_*.csv` and writes summaries to `reports_2024_window_full/hyperparameter_tuning_from_single_csv.xlsx` and `cross_validation_for_slide.xlsx`.

## Quickstart
1) **Environment** (Python 3.10+ recommended):  
   ```bash
   pip install ftfy ekphrasis emoji textblob scikit-learn imbalanced-learn joblib matplotlib tqdm pandas numpy \
       torch torchvision torchaudio transformers sentencepiece
   ```  
   Adjust ekphrasis/torch/transformers per OS/GPU.

2) **Place raw data** `reddit_opinion_PSE_ISR.csv` in repo root **(same folder as notebooks)**. All notebooks assume that relative path and write outputs next to them.

3) **Run notebooks sequentially**: 00 -> 01 -> 02 (label 2024+2025) -> 03 (train) -> 04 (inference) -> 05 (deep learning, optional GPU).  
   - Edit input/output filenames in the config cells if different.  
   - Use default `chunksize` for memory; adjust as needed.

4) **Use artefacts**:  
   - Quick 2025 inference: use `artefacts_2024_window_full/sentiment_pipeline_2024win_stack_count.pkl` via notebook 04.  
   - Deep learning models live in `artefacts_2024_window_deep/` (CUDA needed to load/retrain).

## Notes
- Notebooks use limited parallelism; set env vars `OMP_NUM_THREADS`, etc. (see notebook 04) if you need to cap threads.
- If slang/emoji lexicons are missing, the pipeline falls back to a built-in mini lexicon.
- Required columns: `comment_id, created_time, self_text, score, subreddit`. Preprocess drops empty/URL-only rows.
- All outputs/artefacts are assumed to live in this directory; avoid moving them while notebooks run to prevent path errors.
