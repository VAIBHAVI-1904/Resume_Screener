# ── STEP 1: Import libraries ──────────────────────────────────────────
import streamlit as st
import pickle
import re
import fitz

# ── STEP 2: Load the saved model and vectorizer ───────────────────────
@st.cache_resource
def load_model():
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_model()

# ── STEP 3: Clean text function ───────────────────────────────────────
def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# ── STEP 4: Extract text from PDF ────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ── STEP 5: Build the UI ──────────────────────────────────────────────
st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="centered")
st.title("📄 Resume Screener")
st.markdown("Upload your resume and find out which job role it matches best.")
st.divider()

# ── STEP 6: File uploader ─────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload your resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    analyse = st.button("Analyse Resume")
    if analyse:
        try:
            with st.spinner("Analysing your resume..."):

                # Extract and clean text
                raw_text = extract_text_from_pdf(uploaded_file)

                if not raw_text.strip():
                    st.error("Could not extract text from this PDF. Please try another file.")
                    st.stop()

                cleaned = clean_text(raw_text)

                # Vectorize
                vectorized = tfidf.transform([cleaned])

                # Predict
                prediction = model.predict(vectorized)[0]
                probabilities = model.predict_proba(vectorized)[0]
                confidence = max(probabilities) * 100

            # ── STEP 7: Display results ───────────────────────────────
            st.success("Analysis Complete!")
            st.divider()

            with col1:
                st.markdown("**Predicted Job Role**")
                st.markdown(f"### {prediction}")
            with col2:
                st.markdown("**Match Score**")
                st.markdown(f"### {confidence:.1f}%")

            st.divider()

            # Top 3 matches
            st.subheader("Top 3 Matching Roles")
            classes = model.classes_
            top3_indices = probabilities.argsort()[::-1][:3]

            for i, idx in enumerate(top3_indices):
                role = classes[idx]
                score = probabilities[idx] * 100
                st.progress(int(score), text=f"{i+1}. {role} — {score:.1f}%")

            with st.expander("See extracted resume text"):
                st.text(raw_text[:3000])

        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()