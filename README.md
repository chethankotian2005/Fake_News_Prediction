# Fake News Detector

A machine learning-based web application that can classify news articles as either real or fake using natural language processing and machine learning techniques.

## Features

- **Real-time Classification**: Instantly classify news articles as real or fake
- **Confidence Scores**: See how confident the model is in its predictions
- **Detailed Probabilities**: View the breakdown of fake vs real probabilities
- **User-friendly Interface**: Clean and intuitive Streamlit web interface

## How It Works

The application uses:
- **TF-IDF Vectorization**: Converts text into numerical features
- **Logistic Regression**: Machine learning model trained on thousands of real and fake news articles
- **Text Preprocessing**: Cleans and normalizes input text for better classification

## Local Development

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd fake-news-detector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run Fake_news_prediction.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## Deployment

### Streamlit Cloud

1. **Push your code to GitHub**
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Sign in with GitHub**
4. **Click "New app"**
5. **Select your repository**
6. **Set the main file path**: `Fake_news_prediction.py`
7. **Click "Deploy"**

### Files Structure

```
├── Fake_news_prediction.py    # Main Streamlit application
├── model.pkl                  # Trained machine learning model
├── vectorizer.pkl            # TF-IDF vectorizer
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── Dataset/                  # Training data (not needed for deployment)
    ├── Fake.csv
    └── True.csv
```

## Usage

1. **Enter a news article** in the text area
2. **Click "Predict"** to classify the article
3. **View the results**:
   - Prediction (True ✅ or Fake ❌)
   - Confidence percentage
   - Detailed probabilities

## Model Performance

The model has been trained on a large dataset of real and fake news articles and achieves good accuracy in distinguishing between legitimate news sources and fabricated content.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the [MIT License](LICENSE). 