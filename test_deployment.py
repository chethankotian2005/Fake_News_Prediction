import pickle
import streamlit as st

def test_model():
    """Test if the model and vectorizer can be loaded successfully"""
    try:
        # Test loading model
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        print("✓ Model loaded successfully")
        
        # Test loading vectorizer
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        print("✓ Vectorizer loaded successfully")
        
        # Test prediction
        test_text = "This is a test article about science and technology."
        X_test = vectorizer.transform([test_text])
        prediction = model.predict(X_test)[0]
        probabilities = model.predict_proba(X_test)[0]
        
        print(f"✓ Test prediction successful: {prediction}")
        print(f"✓ Probabilities: {probabilities}")
        print("✓ All tests passed! Ready for deployment.")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing deployment files...")
    success = test_model()
    
    if success:
        print("\n🎉 Your app is ready for Streamlit Cloud deployment!")
        print("\nNext steps:")
        print("1. Push your code to GitHub")
        print("2. Go to https://share.streamlit.io")
        print("3. Deploy your app")
    else:
        print("\n❌ There are issues that need to be fixed before deployment.") 