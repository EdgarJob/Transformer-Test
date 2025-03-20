from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch
import json
from datetime import datetime

app = Flask(__name__)

# Initialize the models
print("Loading models (this might take a few moments)...")
device = 0 if torch.cuda.is_available() else -1

# Sentiment Analysis
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device
)

# Language Detection
language_detector = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device=device
)

# Emotion Analysis
emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device
)

print("All models loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400

        # Analyze sentiment
        sentiment_result = sentiment_analyzer(text)
        
        # Detect language
        language_result = language_detector(text)
        
        # Analyze emotion
        emotion_result = emotion_analyzer(text)
        
        # Format the response
        response = {
            'text': text,
            'sentiment': sentiment_result[0]['label'],
            'sentiment_score': float(sentiment_result[0]['score']),
            'language': language_result[0]['label'],
            'language_score': float(language_result[0]['score']),
            'emotion': emotion_result[0]['label'],
            'emotion_score': float(emotion_result[0]['score']),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save to file
        try:
            with open('analysis_results.json', 'a') as f:
                json.dump(response, f)
                f.write('\n')
        except Exception as e:
            print(f"Error saving results: {e}")

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 