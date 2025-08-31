import streamlit as st
import pickle
import re
import string
import os

# Set page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Simplified preprocessing function (without NLTK dependencies)
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    return text

# Load saved model and TF-IDF vectorizer
@st.cache_resource
def load_model():
    """Load the model and vectorizer with caching"""
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
    st.info("💡 **Tip**: The model works best with longer articles (100+ words)")
    
    # Input area
    user_input = st.text_area(
        "News Article Text",
        height=200,
        placeholder="Paste your news article here..."
    )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔍 Predict", type="primary", use_container_width=True)
    
    if predict_button:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to predict!")
        else:
            try:
                # Preprocess the text
                cleaned_input = clean_text(user_input)
                X_input = vectorizer.transform([cleaned_input])

                # Make prediction
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Prediction result
                if prediction == 1:
                    st.success("✅ **Prediction: REAL NEWS**")
                else:
                    st.error("❌ **Prediction: FAKE NEWS**")
                
                # Confidence and probabilities
                confidence = max(probabilities) * 100
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col2:
                    if prediction == 1:
                        st.metric("Real News Probability", f"{probabilities[1]*100:.1f}%")
                    else:
                        st.metric("Fake News Probability", f"{probabilities[0]*100:.1f}%")
                
                # Detailed probabilities
                st.markdown("**Detailed Probabilities:**")
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.write(f"🎭 Fake News: {probabilities[0]*100:.1f}%")
                with prob_col2:
                    st.write(f"✅ Real News: {probabilities[1]*100:.1f}%")
                
                # Confidence indicator
                if confidence >= 80:
                    st.success("🎯 High confidence prediction")
                elif confidence >= 60:
                    st.warning("⚠️ Medium confidence prediction")
                else:
                    st.info("🤔 Low confidence prediction - consider providing more text")
                    
            except Exception as e:
                st.error(f"❌ Error during prediction: {e}")
    
    # Sample articles for testing
    st.markdown("---")
    with st.expander("🧪 Sample Articles for Testing"):
        st.write("Try these sample articles to test the model:")
        
        sample_articles = {
            "Real News": "NASA's Perseverance rover successfully landed on Mars on February 18, 2021, beginning its mission to search for signs of ancient microbial life. The rover will collect rock and soil samples for future return to Earth and test new technologies for future human exploration of the Red Planet.",
            "Fake News": "BREAKING: Scientists discover that drinking hot water with lemon every morning can cure cancer in just 7 days! The medical establishment has been hiding this simple cure for decades. Share this before it gets deleted!"
        }
        
        for category, article in sample_articles.items():
            st.write(f"**{category}:**")
            st.code(article)
            st.write("")

if __name__ == "__main__":
    main()
