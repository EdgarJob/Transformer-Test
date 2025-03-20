# Sentiment Analysis Tool

A Python-based sentiment analysis tool that uses BERT (Bidirectional Encoder Representations from Transformers) to analyze the sentiment of text input. This tool provides a user-friendly interface for analyzing text sentiment and saving the results.

## Features

- Interactive command-line interface
- Sentiment analysis using BERT model
- Support for multiple languages
- Results saving in JSON format
- GPU acceleration support (if available)
- Detailed sentiment scores and confidence levels

## Requirements

- Python 3.x
- transformers>=4.36.0
- torch>=2.0.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/transformer-test.git
cd transformer-test
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
```bash
python main.py
```

2. Enter your text when prompted. The tool will analyze the sentiment and display:
   - The input text
   - The sentiment rating (1-5 stars)
   - A confidence score (0-1)

3. Type 'quit' to exit the program

4. When exiting, you'll be prompted to save the results to a JSON file

## Example Output

```
Using device: CPU

Loading BERT model (this might take a few moments)...
BERT model loaded successfully!

Welcome to the Sentiment Analysis Tool!
Enter your text (or 'quit' to exit):

Enter text: I love this amazing product!

Sentiment Analysis Result:
Text: 'I love this amazing product!'
Sentiment: 5 stars
Confidence Score: 0.9876
```

## Model Details

This tool uses the `nlptown/bert-base-multilingual-uncased-sentiment` model, which:
- Is based on BERT architecture
- Supports multiple languages
- Provides 5-star sentiment ratings
- Is fine-tuned for sentiment analysis

## File Structure

- `main.py`: Main script containing the sentiment analysis implementation
- `requirements.txt`: List of Python dependencies
- `*.json`: Generated result files (created when saving results)

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [BERT Model](https://github.com/google-research/bert)
- [nlptown](https://huggingface.co/nlptown) for the sentiment analysis model 