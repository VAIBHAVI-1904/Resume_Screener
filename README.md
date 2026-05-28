---
title: Resume Screener
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 📄 Resume Screener

An NLP-based web app that predicts which job role a resume best fits, built using machine learning and deployed on Hugging Face Spaces.

## 🔗 Live Demo
[Try it here](https://huggingface.co/spaces/Vaibhavi-1904/Resume_Screener)

## 🧠 How it works
1. User uploads a resume as a PDF
2. Text is extracted and cleaned
3. TF-IDF converts the text into numerical features
4. A Linear SVC model predicts the most suitable job role
5. Top 3 matching roles are displayed with scores

## 📊 Model Performance
- **Algorithm:** TF-IDF + Linear SVC (with CalibratedClassifierCV)
- **Dataset:** 2,484 resumes across 24 job categories
- **Accuracy:** 67.61%

## 🛠️ Tech Stack
| Tool | Purpose |
|---|---|
| scikit-learn | TF-IDF vectorization + Linear SVC model |
| Streamlit | Web UI |
| PyMuPDF | PDF text extraction |
| Hugging Face Spaces | Free cloud deployment |

## 📁 Project Structure
resume_screener/
├── app.py           # Streamlit web app
├── train.py         # Model training script
├── model/
│   ├── model.pkl    # Trained model
│   └── tfidf.pkl    # TF-IDF vectorizer
├── requirements.txt
└── Dockerfile

## 🚀 Run Locally
'''bash'''

git clone https://github.com/Vaibhavi-1904/Resume_Screener.git
cd Resume_Screener
pip install -r requirements.txt
streamlit run app.py


