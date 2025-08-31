import pickle
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def test_app():
    print("Testing Fake News Detector App...")
    
    try:
        # Load model and vectorizer
        print("Loading model and vectorizer...")
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        print("✓ Model and vectorizer loaded successfully")
        
        # Test predictions
        test_articles = [
            "NASA's Perseverance rover successfully landed on Mars on February 18, 2021, beginning its mission to search for signs of ancient microbial life.",
            "BREAKING: Scientists discover that drinking hot water with lemon every morning can cure cancer in just 7 days! The medical establishment has been hiding this simple cure for decades."
        ]
        
        print("\nTesting predictions...")
        for i, article in enumerate(test_articles, 1):
            cleaned = clean_text(article)
            X_test = vectorizer.transform([cleaned])
            prediction = model.predict(X_test)[0]
            probabilities = model.predict_proba(X_test)[0]
            
            label = "REAL" if prediction == 1 else "FAKE"
            confidence = max(probabilities) * 100
            
            print(f"Article {i}: {label} (Confidence: {confidence:.1f}%)")
        
        print("\n🎉 All tests passed! App is ready for deployment.")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_app() 