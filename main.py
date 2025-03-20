# Import required libraries
from transformers import pipeline  # Imports the pipeline class from Hugging Face's transformers library
import torch  # PyTorch library for deep learning
import json  # For handling JSON file operations
from datetime import datetime  # For generating timestamps

def save_results(results, filename=None):
    """
    Save analysis results to a JSON file
    Args:
        results: List of sentiment analysis results
        filename: Optional custom filename, if None will generate one with timestamp
    """
    if filename is None:
        # Creates a filename with current date and time if none provided
        filename = f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Writes results to a JSON file with nice formatting (indent=4)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {filename}")

def analyze_sentiment(text, analyzer):
    """
    Analyze sentiment of a single text
    Args:
        text: The input text to analyze
        analyzer: The sentiment analysis pipeline
    Returns:
        Dictionary containing the text, sentiment label, and confidence score
    """
    # Runs the sentiment analysis on the input text
    result = analyzer(text)
    # Returns a dictionary with the text, sentiment label, and confidence score
    return {
        "text": text,
        "sentiment": result[0]["label"],
        "score": float(result[0]["score"])
    }

def main():
    # Check if GPU (CUDA) is available, uses CPU if not
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")
    
    # Create the sentiment analysis pipeline
    print("\nLoading BERT model (this might take a few moments)...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",  # Specifies the task type
        model="nlptown/bert-base-multilingual-uncased-sentiment",  # The specific model to use
        device=device  # Whether to use GPU or CPU
    )
    
    # Welcome message and instructions
    print("\nBERT model loaded successfully!")
    print("\nWelcome to the Sentiment Analysis Tool!")
    print("Enter your text (or 'quit' to exit):")
    
    # List to store all analysis results
    results = []
    
    # Main program loop
    while True:
        # Get input from user
        text = input("\nEnter text: ").strip()
        
        # Check if user wants to quit
        if text.lower() == 'quit':
            break
            
        # Skip empty inputs
        if not text:
            print("Please enter some text!")
            continue
            
        # Analyze the sentiment
        result = analyze_sentiment(text, sentiment_analyzer)
        results.append(result)  # Store the result
        
        # Display the analysis result
        print(f"\nSentiment Analysis Result:")
        print(f"Text: '{result['text']}'")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence Score: {result['score']:.4f}")
    
    # After the loop ends, ask if user wants to save results
    if results:
        save_choice = input("\nWould you like to save the results? (y/n): ").strip().lower()
        if save_choice == 'y':
            save_results(results)
    
    print("\nThank you for using the Sentiment Analysis Tool!")

# Program entry point - only runs if the script is executed directly
if __name__ == "__main__":
    main()

