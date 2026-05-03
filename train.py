# ── STEP 1: Import libraries ──────────────────────────────────────────
import pandas as pd
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── STEP 2: Load the dataset ──────────────────────────────────────────
df = pd.read_csv("data/Resume.csv")

# ── STEP 3: Clean the resume text ─────────────────────────────────────
def clean_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

df["cleaned"] = df["Resume_str"].apply(clean_text)

# ── STEP 4: Define input and output ───────────────────────────────────
X = df["cleaned"]
y = df["Category"]

# ── STEP 5: Split into train and test sets ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── STEP 6: TF-IDF Vectorizer ─────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ── STEP 7: Train the model ───────────────────────────────────────────
svc = LinearSVC(max_iter=1000, C=1.0, class_weight='balanced')
model = CalibratedClassifierCV(svc)
model.fit(X_train_tfidf, y_train)

# ── STEP 8: Test accuracy ─────────────────────────────────────────────
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ── STEP 9: Save model and vectorizer ─────────────────────────────────
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved successfully!")