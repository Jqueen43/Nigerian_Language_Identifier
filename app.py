"""
Nigerian Language Identifier — Streamlit App
=============================================
Classifies text as English, Hausa, Igbo, or Yoruba using the TF-IDF
character n-gram vectorizer + Linear SVM model trained in the
accompanying notebook.

Expects two files saved by the notebook, in the same folder as this
script (or upload them from the sidebar if they live elsewhere):
    - linear_svm_model.joblib
    - tfidf_vectorizer.joblib

Run with:
    streamlit run app.py
"""

import re
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
    page_title="Nigerian Language Identifier",
    page_icon="🗣️",
    layout="centered",
)

MODEL_PATH = "linear_svm_model.joblib"
VECTORIZER_PATH = "tfidf_vectorizer.joblib"

# Same character-cleaning rule used in the notebook's preprocessing step,
# so text typed here gets the same treatment the training data received.
ALLOWED_PATTERN = (
    r"[^a-zA-Z0-9\s.,!?\'\"\-"
    r"\u00C0-\u017F"   # Latin-1 Supplement & Extended-A (accents)
    r"\u0180-\u02AF"   # Hausa hooked letters: ɓ ɗ ƙ ƴ
    r"\u1E00-\u1EFF"   # Yoruba/Igbo dotted letters: ẹ ọ ṣ ị ụ ṅ
    r"\u0300-\u036F"   # combining diacritical / tone marks
    r"]"
)

LANGUAGE_LABELS = {
    "english": "English 🇬🇧",
    "hausa": "Hausa",
    "igbo": "Igbo",
    "yoruba": "Yoruba",
}


def clean_text(text: str) -> str:
    return re.sub(ALLOWED_PATTERN, "", str(text))


@st.cache_resource(show_spinner=False)
def load_artifacts(model_file, vectorizer_file):
    model = joblib.load("linear_svm_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib)
    return model, vectorizer


def softmax(scores: np.ndarray) -> np.ndarray:
    """Turn LinearSVC's decision_function margins into pseudo-probabilities.

    LinearSVC has no predict_proba, so this is only an approximate
    confidence score for display purposes — not a calibrated probability.
    """
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def predict(texts, model, vectorizer):
    cleaned = [clean_text(t) for t in texts]
    features = vectorizer.transform(cleaned)
    predictions = model.predict(features)
    margins = model.decision_function(features)
    if margins.ndim == 1:
        # Binary edge case: decision_function returns a 1-D array.
        margins = np.column_stack([-margins, margins])
    probs = softmax(margins)
    confidences = probs.max(axis=1) * 100
    return predictions, confidences


# ---------------------------------------------------------------------
# Sidebar — load model artifacts
# ---------------------------------------------------------------------
st.sidebar.header("Model files")
st.sidebar.caption(
    "By default the app looks for `linear_svm_model.joblib` and "
    "`tfidf_vectorizer.joblib` next to this script. Upload them here "
    "instead if they're stored elsewhere."
)
uploaded_model = st.sidebar.file_uploader("Model (.joblib)", type="joblib")
uploaded_vectorizer = st.sidebar.file_uploader("Vectorizer (.joblib)", type="joblib")

model_source = uploaded_model if uploaded_model is not None else MODEL_PATH
vectorizer_source = uploaded_vectorizer if uploaded_vectorizer is not None else VECTORIZER_PATH

try:
    model, vectorizer = load_artifacts(model_source, vectorizer_source)
    model_ready = True
except FileNotFoundError:
    model_ready = False
    st.sidebar.error(
        "Couldn't find the model/vectorizer files. Place them next to "
        "app.py, or upload them above."
    )
except Exception as e:
    model_ready = False
    st.sidebar.error(f"Couldn't load the model files: {e}")

# ---------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------
st.title("🗣️ Nigerian Language Identifier")
st.write(
    "Detects whether a piece of text is **English**, **Hausa**, **Igbo**, "
    "or **Yoruba**, using a Linear SVM trained on character-level TF-IDF "
    "features (NaijaVoices dataset + an English sentence dataset)."
)

tab_single, tab_batch = st.tabs(["Single text", "Batch (CSV upload)"])

with tab_single:
    default_examples = [
        "Type or paste text here…",
        "Ina son in koyi yadda ake dafa abinci.",
        "Nne m na-esi nri ụtọ nke ukwuu.",
        "Báwo ni nǹkan ṣe ń lọ ní ilé ẹ̀kọ́ rẹ?",
        "The quick brown dog jumps over the lazy cat.",
    ]
    example_choice = st.selectbox(
        "Try an example, or choose the first option to type your own:",
        default_examples,
    )
    text_input = st.text_area(
        "Text to classify",
        value="" if example_choice == default_examples[0] else example_choice,
        height=120,
        placeholder="e.g. Báwo ni nǹkan ṣe ń lọ?",
    )

    if st.button("Identify language", type="primary", disabled=not model_ready):
        if not text_input.strip():
            st.warning("Please enter some text first.")
        else:
            preds, confs = predict([text_input], model, vectorizer)
            label = preds[0]
            display_label = LANGUAGE_LABELS.get(label.lower(), label)
            st.success(f"**Predicted language:** {display_label}")
            st.metric("Confidence (approximate)", f"{confs[0]:.1f}%")
            st.caption(
                "Confidence is derived from the SVM's decision margins via "
                "softmax — it's a rough indicator, not a calibrated probability."
            )

with tab_batch:
    st.write("Upload a CSV with a column of text to classify each row.")
    csv_file = st.file_uploader("CSV file", type="csv", key="batch_csv")
    if csv_file is not None:
        df_batch = pd.read_csv(csv_file)
        text_column = st.selectbox("Which column contains the text?", df_batch.columns)
        if st.button("Run batch prediction", disabled=not model_ready):
            preds, confs = predict(df_batch[text_column].astype(str).tolist(), model, vectorizer)
            df_batch["predicted_language"] = preds
            df_batch["confidence_%"] = confs.round(1)
            st.dataframe(df_batch)
            st.download_button(
                "Download results as CSV",
                df_batch.to_csv(index=False).encode("utf-8"),
                file_name="language_predictions.csv",
                mime="text/csv",
            )

st.divider()
st.caption(
    "Model: Linear SVM · Features: character n-grams (2–4) TF-IDF · "
    "Trained on NaijaVoices (Hausa/Igbo/Yoruba) + expanded English sentences."
)
