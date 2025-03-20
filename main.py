from transformers import pipeline
import torch

def main():
    # Check if CUDA is available
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'CUDA' if device == 0 else 'CPU'}")
    
    # Create a sentiment analysis pipeline using distilbert
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    
    # Example texts for demonstration
    example_texts = [
        "I love using transformers library, it's amazing!",
        "This code is not working as expected.",
        "The weather today is quite pleasant.",
        "I'm really disappointed with the service."
    ]
    
    # Analyze sentiment for each example text
    print("\nSentiment Analysis Results:")
    for text in example_texts:
        result = sentiment_analyzer(text)
        print(f"Text: '{text}'")
        print(f"Sentiment: {result[0]['label']}, Score: {result[0]['score']:.4f}\n")
    
    # Example of text classification with a different model
    print("Creating a text classification pipeline...")
    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )
    
    # Sample classification
    classification_text = "Transformers library makes NLP tasks simple and efficient."
    classification_result = classifier(classification_text)
    print(f"Classification for: '{classification_text}'")
    print(f"Result: {classification_result[0]['label']}, Score: {classification_result[0]['score']:.4f}")


if __name__ == "__main__":
    print("Starting text classification demo with transformers library...")
    main()
    print("Demo completed!")

