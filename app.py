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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    return ' '.join([page.extract_text() for page in reader.pages])

# Initialize models
print("Loading models (this might take a few moments)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use {device}")

# Existing models
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
language_detector = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection", device=device)
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=device)

# New models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
ner_analyzer = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", device=device)
topic_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
keyword_extractor = pipeline("feature-extraction", model="distilbert-base-uncased", device=device)

print("All models loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Please provide some text to analyze'}), 400

        # Existing analysis
        sentiment_result = sentiment_analyzer(text)[0]
        language_result = language_detector(text)[0]
        emotion_result = emotion_analyzer(text)[0]

        # New analysis features
        # 1. Text Summarization (for longer texts)
        summary = ""
        if len(text.split()) > 50:  # Only summarize longer texts
            summary_result = summarizer(text, max_length=130, min_length=30, do_sample=False)
            summary = summary_result[0]['summary_text']

        # 2. Named Entity Recognition
        ner_results = ner_analyzer(text)
        entities = {}
        for entity in ner_results:
            entity_type = entity['entity']
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity['word'])

        # 3. Topic Classification
        topics = ["technology", "business", "sports", "entertainment", "science", "health", "politics"]
        topic_result = topic_classifier(text, topics, multi_label=True)
        topic_scores = dict(zip(topic_result['labels'], topic_result['scores']))

        # 4. Keyword Extraction (using feature extraction)
        keyword_features = keyword_extractor(text, return_tensors="pt")
        # Get the mean of the last hidden state as a simple keyword representation
        keyword_embedding = keyword_features[0].mean(dim=1).squeeze().tolist()

        # Prepare response
        response = {
            'text': text,
            'sentiment': sentiment_result['label'],
            'sentiment_score': sentiment_result['score'],
            'language': language_result['label'],
            'language_score': language_result['score'],
            'emotion': emotion_result['label'],
            'emotion_score': emotion_result['score'],
            'summary': summary,
            'entities': entities,
            'topics': topic_scores,
            'keyword_embedding': keyword_embedding,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save results
        try:
            with open('analysis_results.json', 'a') as f:
                f.write(json.dumps(response) + '\n')
        except Exception as e:
            print(f"Error saving results: {e}")

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
            return analyze_text(text)
            
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_text(text):
    try:
        # Existing analysis
        sentiment_result = sentiment_analyzer(text)[0]
        language_result = language_detector(text)[0]
        emotion_result = emotion_analyzer(text)[0]

        # New analysis features
        # 1. Text Summarization (for longer texts)
        summary = ""
        if len(text.split()) > 50:  # Only summarize longer texts
            summary_result = summarizer(text, max_length=130, min_length=30, do_sample=False)
            summary = summary_result[0]['summary_text']

        # 2. Named Entity Recognition
        ner_results = ner_analyzer(text)
        entities = {}
        for entity in ner_results:
            entity_type = entity['entity']
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity['word'])

        # 3. Topic Classification
        topics = ["technology", "business", "sports", "entertainment", "science", "health", "politics"]
        topic_result = topic_classifier(text, topics, multi_label=True)
        topic_scores = dict(zip(topic_result['labels'], topic_result['scores']))

        # 4. Keyword Extraction (using feature extraction)
        keyword_features = keyword_extractor(text, return_tensors="pt")
        # Get the mean of the last hidden state as a simple keyword representation
        keyword_embedding = keyword_features[0].mean(dim=1).squeeze().tolist()

        # Prepare response
        response = {
            'text': text,
            'sentiment': sentiment_result['label'],
            'sentiment_score': sentiment_result['score'],
            'language': language_result['label'],
            'language_score': language_result['score'],
            'emotion': emotion_result['label'],
            'emotion_score': emotion_result['score'],
            'summary': summary,
            'entities': entities,
            'topics': topic_scores,
            'keyword_embedding': keyword_embedding,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Save results
        try:
            with open('analysis_results.json', 'a') as f:
                f.write(json.dumps(response) + '\n')
        except Exception as e:
            print(f"Error saving results: {e}")

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002) 