import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("spam.csv")
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

data = load_data()

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model(data):
    X = data['message']
    y = data['label']

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    model = SVC(kernel='linear')
    model.fit(X_vectorized, y)

    return model, vectorizer

model, vectorizer = train_model(data)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📧 Email Spam Classifier (SVM)")
st.write("This app classifies an email message as Spam or Not Spam using Machine Learning.")

# Input box
email_input = st.text_area("Enter Email Message")

# Predict button
if st.button("Classify"):
    if email_input.strip() == "":
        st.warning("⚠️ Please enter a message")
    else:
        # Transform input
        input_vector = vectorizer.transform([email_input])

        # Predict
        prediction = model.predict(input_vector)

        # Output
        if prediction[0] == 1:
            st.error("🚫 This is a SPAM message")
        else:
            st.success("✅ This is NOT SPAM")

# -----------------------------
# Optional: Show Dataset
# -----------------------------
if st.checkbox("Show Dataset"):
    st.write(data)