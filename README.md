# Israel-Palestine Sentiment Analysis (Reddit 2024-2025)

Analisis sentimen komentar Reddit terkait konflik Israel-Palestina untuk periode 2024-2025. Proyek ini memfilter komentar (tgl 1-10 tiap bulan), membersihkan teks, memberi label otomatis dengan TextBlob, melatih model klasik (Count/TF-IDF + NB/SVM/LR + stacking), menguji model deep learning (BiLSTM, BERT, stacking), serta melakukan inferensi dan rangkuman tren 2025.

> Dataset **tidak disertakan** di repo karena besar. Unduh dari Google Drive: https://drive.google.com/drive/u/0/folders/1VlPXCwDXPtolbvzqpMy0eLGctCZ11I_o dan letakkan file mentahnya di direktori yang sama dengan notebook sebelum menjalankan apa pun.

## Struktur & Artefak Penting
- `00_FilterData-Chunk.ipynb` - filter window tanggal 1-10 per bulan dari file mentah.
- `01_PreProcessing.ipynb` - pembersihan + normalisasi + stopword + stemming, hasil `..._window_clean.csv`.
- `02_Labeling_TextBlob.ipynb` - labeling otomatis (TextBlob) -> `..._window_labeled_textblob.csv`.
- `03_Training-Tuning-Stacking2024.ipynb` - model klasik, CV/tuning, simpan pipeline terbaik.
- `04_Inference_2025.ipynb` - muat pipeline terbaik, prediksi 2025, timeseries.
- `05_DeepLearning_BiLSTM_BERT_Stacking_FIX.ipynb` - BiLSTM, BERT, dan stacking ensemble (butuh GPU).
- `CV-RingkasanFold-ClassicalML.ipynb` - rangkum hasil CV ke Excel (`reports_2024_window_full/*.xlsx`).
- Artefak model:
  - (Disimpan di Google Drive karena ukuran besar) folder `artefacts_2024_window_full/` berisi pipeline klasik terbaik `sentiment_pipeline_2024win_stack_count.pkl` + `model_meta.json`.
  - (Disimpan di Google Drive) folder `artefacts_2024_window_deep/` untuk model BiLSTM/BERT/stacking + hasil.
  - (Disimpan di Google Drive) folder `reports_2024_window_full/` untuk ringkasan Excel CV.

## Alur Data
Semua keluaran ditulis di folder yang sama dengan notebook.

1. **Raw**: letakkan `reddit_opinion_PSE_ISR.csv` (gabungan 2024+2025) di root repo.
2. **Filter window (00)**  
   - Membaca kolom `comment_id, created_time, self_text, score, subreddit`.  
   - Stream `chunksize=1_000_000`, parse datetime, ambil hari 1-10, split 2024 vs 2025, dedup ringan per-tahun (`comment_id`).  
   - Output: `reddit_opinion_PSE_ISR_2024_window.csv`, `reddit_opinion_PSE_ISR_2025_window.csv`.
3. **Preprocess (01)**  
   - Dependensi: `ftfy, ekphrasis, emoji, scikit-learn, nltk (stemmer/tokenizer)`.  
   - Pipeline: cleaning URL/HTML/mention/non-print + lower + social normalize (ekphrasis + emoji map) + slang replace (lexicon opsional, fallback bawaan) + tokenisasi + stopword removal + Porter stemming + gabung `final_text`; kolom `month` diisi dari `created_time`.  
   - Output: `..._window_clean.csv` (2024 & 2025).
4. **Labeling (02)**  
   - Dependensi: `textblob`.  
   - Threshold polaritas: `>= +0.05 = Positif`, `<= -0.05 = Negatif`, sisanya Netral.  
   - Output: `reddit_opinion_PSE_ISR_2024_window_labeled_textblob.csv` dan `reddit_opinion_PSE_ISR_2025_window_labeled_textblob.csv` (plus ringkasan distribusi/mean polarity di notebook).
5. **Training klasik (03)**  
   - Train: Jan-Okt 2024, Test: Nov-Des 2024 (kolom `month`).  
   - Kandidat: Count/TF-IDF + ComplementNB, LinearSVC, Logistic Regression; Stacking (NB + Calibrated SVM + LR, meta=LR).  
   - Hyperparameter: `RandomizedSearchCV` (F1-macro, `n_jobs=1`), hasil CV tersimpan di `artefacts_2024_window_full/cv_*.csv` dan `cv_results_2024_window.json`.  
   - **Pemenang**: `stack_count` (CountVectorizer 1-gram, min_df=5, max_df=0.9) dengan Calibrated SVM (C=5, max_iter=10000, tol=0.003), NB (alpha=0.5), LR (C=5), meta LR (C=1).  
   - Skor utama (hold-out Nov-Des): Accuracy 0.9528, Macro F1 0.9525 (`artefacts_2024_window_full/model_meta.json`). CV terbaik: 0.9486.  
   - Pipeline disimpan ke `sentiment_pipeline_2024win_stack_count.pkl` + meta JSON (label order, text column, dsb.).
6. **Inference 2025 (04)**  
   - Memilih pipeline terbaik via `model_meta.json`/`cv_results_2024_window.json` lalu memuat `.joblib/.pkl`.  
   - Input: `reddit_opinion_PSE_ISR_2025_window_clean.csv`.  
   - Output: prediksi ke `reddit_opinion_PSE_ISR_2025_window_pred.csv`; agregasi bulanan counts (`sentiment_timeseries_2025_count.csv`) & share (`..._share.csv`); plot timeseries ditampilkan inline (PNG opsional).
7. **Deep learning (05)** - GPU wajib  
   - Split sama (Jan-Okt train, Nov-Des test).  
   - **BiLSTM** (CountVectorizer vocab + seq, max_len=128, embed_dim=128, hidden=256, dropout=0.3, epochs=5).  
     - Hasil: Accuracy 0.9797, F1 0.9798 (`bilstm_results.json`), model `bilstm_bow_model.pt`.  
   - **BERT base uncased** (max_len=128, bs=16, epochs=3, lr=2e-5).  
     - Hasil: Accuracy 0.9856, F1 0.9857 (`bert_results.json`), artefak `bert_model/`, `bert_tokenizer/`.  
   - **Stacking** (prob BiLSTM + BERT -> Logistic Regression).  
     - Hasil: Accuracy 0.9841, F1 0.9842 (`stacking_results.json`), meta `stacking_meta_lr.joblib`, proba cache `stacking_proba_*.npz`.
8. **CV Ringkasan**  
   - `CV-RingkasanFold-ClassicalML.ipynb` membaca `cv_*.csv` lalu menulis ringkasannya ke `reports_2024_window_full/hyperparameter_tuning_from_single_csv.xlsx` dan `cross_validation_for_slide.xlsx`.

## Menjalankan Secara Singkat
1) **Persiapan lingkungan** (Python 3.10+ disarankan):  
   ```bash
   pip install ftfy ekphrasis emoji textblob scikit-learn imbalanced-learn joblib matplotlib tqdm pandas numpy \
       torch torchvision torchaudio transformers sentencepiece
   ```  
   Untuk ekphrasis/torch/transformers, pastikan sesuai GPU/OS.

2) **Taruh data mentah** `reddit_opinion_PSE_ISR.csv` di root repo **(direktori yang sama dengan notebook)**. Semua notebook mengasumsikan path relatif tersebut dan akan menulis output di lokasi yang sama.

3) **Jalankan notebook berurutan**: 00 -> 01 -> 02 (labeling 2024+2025) -> 03 (training) -> 04 (inference) -> 05 (deep learning, opsional GPU).  
   - Ubah nama file input/output di sel konfigurasi jika berbeda.  
   - Gunakan `chunksize` default untuk hemat RAM; sesuaikan bila perlu.

4) **Gunakan artefak**:  
   - Prediksi cepat 2025 dapat langsung memakai `artefacts_2024_window_full/sentiment_pipeline_2024win_stack_count.pkl` via notebook 04.  
   - Model deep learning siap pakai di `artefacts_2024_window_deep/` (perlu CUDA untuk load/train ulang).

## Catatan
- Notebook mem-paralel sebagian kecil; set variabel lingkungan `OMP_NUM_THREADS`, dll. (lihat notebook 04) bila butuh membatasi thread.
- Bila kamus slang/emoji tidak tersedia, pipeline menggunakan fallback bawaan.
- Pastikan kolom wajib ada: `comment_id, created_time, self_text, score, subreddit`. Preprocess akan menolak baris kosong/berisi URL saja.
- Semua file output dan artefak diasumsikan berada di direktori ini; jangan pindahkan saat notebook sedang dijalankan untuk menghindari path error.
