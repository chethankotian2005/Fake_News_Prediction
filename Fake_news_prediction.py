import streamlit as st
import pickle
import re
import string

# Simplified preprocessing function (without NLTK dependencies)
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    return text

# Load saved model and TF-IDF vectorizer
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    
    # Streamlit app UI
    st.title("Fake News Detector 📰")
    st.write("Enter a news article below to check if it's real or fake:")

    user_input = st.text_area("News Article Text")

    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text to predict!")
        else:
            # Preprocess the text first (same as training)
            cleaned_input = clean_text(user_input)
            X_input = tfidf.transform([cleaned_input])

            # Make prediction
            prediction = model.predict(X_input)[0]
            label = "True ✅" if prediction == 1 else "Fake ❌"

            st.success(f"Prediction → {label}")
            
            # Add confidence score
            confidence = model.predict_proba(X_input)[0]
            st.info(f"Confidence: {max(confidence)*100:.2f}%")
            
            # Show detailed probabilities
            st.write("**Detailed Probabilities:**")
            st.write(f"Fake: {confidence[0]*100:.2f}%")
            st.write(f"True: {confidence[1]*100:.2f}%")

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.write("Please make sure model.pkl and vectorizer.pkl files exist in the current directory.")
