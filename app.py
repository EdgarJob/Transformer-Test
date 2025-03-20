from flask import Flask, render_template, request, jsonify
from transformers import pipeline
import torch
import json
from datetime import datetime
import os
from werkzeug.utils import secure_filename
from docx import Document
from PyPDF2 import PdfReader

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'docx', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

def analyze_text(text):
    # Analyze sentiment
    sentiment_result = sentiment_analyzer(text)
    
    # Detect language
    language_result = language_detector(text)
    
    # Analyze emotion
    emotion_result = emotion_analyzer(text)
    
    return {
        'text': text,
        'sentiment': sentiment_result[0]['label'],
        'sentiment_score': float(sentiment_result[0]['score']),
        'language': language_result[0]['label'],
        'language_score': float(language_result[0]['score']),
        'emotion': emotion_result[0]['label'],
        'emotion_score': float(emotion_result[0]['score']),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text based on file type
            if filename.endswith('.docx'):
                text = extract_text_from_docx(file_path)
            else:  # PDF
                text = extract_text_from_pdf(file_path)
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            # Analyze the extracted text
            response = analyze_text(text)
            
            # Save results
            try:
                with open('analysis_results.json', 'a') as f:
                    json.dump(response, f)
                    f.write('\n')
            except Exception as e:
                print(f"Error saving results: {e}")
            
            return jsonify(response)
        
        return jsonify({'error': 'Invalid file type'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400

        response = analyze_text(text)
        
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