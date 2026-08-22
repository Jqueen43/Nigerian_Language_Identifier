# Nigerian Language Identifier

A text classifier that identifies whether a piece of text is **English**,
**Hausa**, **Igbo**, or **Yoruba**. Built as a capstone project for a
government-run AI/ML training program.

## Overview

- **Data**: [NaijaVoices dataset](https://huggingface.co/datasets/naijavoices/naijavoices-dataset-compressed)
  (Hausa, Igbo, Yoruba text) combined with an English sentence dataset
  ([agentlans/expanded-english-sentences](https://huggingface.co/datasets/agentlans/expanded-english-sentences))
  from Hugging Face. Balanced to ~5,000 samples per language, cleaned of
  noisy/foreign characters while preserving each language's diacritics
  (e.g. Hausa ɓ ɗ ƙ ƴ, Yoruba/Igbo ẹ ọ ṣ ị ụ ṅ).
- **Features**: character-level TF-IDF (n-grams 2–4, `char_wb`).
- **Models compared**: Multinomial Naive Bayes, Logistic Regression,
  Linear SVM (best-performing, used in the final app).
- **Evaluation**: accuracy, per-class classification report, confusion
  matrix, 5-fold cross-validation on the SVM.

## Repo contents

| File | Description |
|---|---|
| `nigerian_language_identifier.ipynb` | Full notebook: data loading, cleaning, EDA, feature engineering, model training/evaluation |
| `README.md` | Project overview and steps taken to execute the project |
| `app.py` | Streamlit demo app for interactive and batch predictions |
| `requirements.txt` | Python dependencies for the app |
| `balanced_multilingual_dataset.csv` | Combined dataset (5,000 samples per language) merging NaijaVoices Hausa/Igbo/Yoruba text with sampled English sentences, before final cleaning |
| `final_cleaned_dataset.csv` | Cleaned, de-duplicated, and balanced dataset used for the train/test split and model training |
| `linear_svm_model.joblib` (`naija-language-classifier-svm`) | Trained Linear SVM classifier — the best-performing model, used by `app.py` |
| `tf-idf_vectorizer.joblib` | Fitted TF-IDF character n-gram vectorizer used to transform raw text into features for the model |

## Running the notebook

Open `nigerian_language_identifier.ipynb` in Google Colab or Jupyter.
It downloads the NaijaVoices dataset from a gated Hugging Face repo, so
you'll need a Hugging Face account/token with access approved.

## Running the app

The app expects `linear_svm_model.joblib` (the `naija-language-classifier-svm`
model) and `tf-idf_vectorizer.joblib` — both saved by the last cells of the
notebook — in the same folder as `app.py`. If you rename either file, update
the `MODEL_PATH` / `VECTORIZER_PATH` constants near the top of `app.py` to
match.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Results

See the notebook for full classification reports and confusion matrices.
The Linear SVM was the best-performing model on the held-out test set.

```

## Results

See the notebook for full classification reports and confusion matrices.
The Linear SVM was the best-performing model on the held-out test set.

