import streamlit as st
import pickle
import re
import string

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Simplified preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Main app
def main():
    st.title("Fake News Detector 📰")
    st.markdown("---")
    
    # Load model
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.error("❌ Failed to load the model. Please check if model.pkl and vectorizer.pkl files exist.")
        st.stop()
    
    # App description
    st.write("Enter a news article below to check if it's real or fake:")
    
    # Input area
    user_input = st.text_area(
        "News Article Text",
        height=200,
        placeholder="Paste your news article here..."
    )
    
    # Prediction button
    if st.button("🔍 Predict", type="primary"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to predict!")
        else:
            try:
                # Preprocess and predict
                cleaned_input = clean_text(user_input)
                X_input = vectorizer.transform([cleaned_input])
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                if prediction == 1:
                    st.success("✅ **Prediction: REAL NEWS**")
                else:
                    st.error("❌ **Prediction: FAKE NEWS**")
                
                # Confidence
                confidence = max(probabilities) * 100
                st.info(f"Confidence: {confidence:.1f}%")
                
                # Probabilities
                st.write("**Probabilities:**")
                st.write(f"Fake News: {probabilities[0]*100:.1f}%")
                st.write(f"Real News: {probabilities[1]*100:.1f}%")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
    
    # Sample articles
    st.markdown("---")
    with st.expander("🧪 Sample Articles for Testing"):
        st.write("**Real News:**")
        st.code("NASA's Perseverance rover successfully landed on Mars on February 18, 2021, beginning its mission to search for signs of ancient microbial life.")
        
        st.write("**Fake News:**")
        st.code("BREAKING: Scientists discover that drinking hot water with lemon every morning can cure cancer in just 7 days! The medical establishment has been hiding this simple cure for decades.")

if __name__ == "__main__":
    main()
