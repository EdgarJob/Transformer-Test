from transformers import pipeline
import torch
import json
from datetime import datetime

def save_results(results, filename=None):
    """Save analysis results to a JSON file"""
    if filename is None:
        filename = f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {filename}")

def analyze_sentiment(text, analyzer):
    """Analyze sentiment of a single text"""
    result = analyzer(text)
    return {
        "text": text,
        "sentiment": result[0]["label"],
        "score": float(result[0]["score"])
    }

def main():
    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")
    
    # Create a sentiment analysis pipeline using BERT
    print("\nLoading BERT model (this might take a few moments)...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device
    )
    
    print("\nBERT model loaded successfully!")
    print("\nWelcome to the Sentiment Analysis Tool!")
    print("Enter your text (or 'quit' to exit):")
    
    results = []
    while True:
        text = input("\nEnter text: ").strip()
        
        if text.lower() == 'quit':
            break
            
        if not text:
            print("Please enter some text!")
            continue
            
        # Analyze sentiment
        result = analyze_sentiment(text, sentiment_analyzer)
        results.append(result)
        
        # Display result
        print(f"\nSentiment Analysis Result:")
        print(f"Text: '{result['text']}'")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence Score: {result['score']:.4f}")
    
    # Save results if any analysis was performed
    if results:
        save_choice = input("\nWould you like to save the results? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_results(results)
    
    print("\nThank you for using the Sentiment Analysis Tool!")

if __name__ == "__main__":
    main()

