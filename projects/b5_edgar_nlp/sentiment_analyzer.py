"""Analyze sentiment using FinBERT AI model"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np


class SentimentAnalyzer:
    """Uses FinBERT to detect financial sentiment"""
    
    def __init__(self):
        print("📥 Loading FinBERT model (takes 30-60 seconds)...")
        
        # Load FinBERT model and tokenizer
        model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Use CPU
        self.device = "cpu"
        self.model.eval()
        
        print("✅ FinBERT loaded successfully")
    
    def analyze(self, text, max_sentences=20):
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            max_sentences: Number of sentences to analyze (default: 20)
        
        Returns:
            Dictionary with sentiment results
        """
        if len(text) < 100:
            return self._empty_result()
        
        # Split into sentences
        sentences = self._split_sentences(text)
        sentences = sentences[:max_sentences]  # Take first N sentences
        
        if not sentences:
            return self._empty_result()
        
        # Analyze each sentence
        results = []
        for sentence in sentences:
            if len(sentence) > 20:  # Skip very short sentences
                sentiment = self._analyze_sentence(sentence)
                results.append(sentiment)
        
        if not results:
            return self._empty_result()
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _analyze_sentence(self, sentence):
        """Analyze a single sentence"""
        # Tokenize (max 512 tokens for FinBERT)
        inputs = self.tokenizer(
            sentence[:500],  # Truncate long sentences
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT labels: 0=positive, 1=negative, 2=neutral
        probs = predictions[0].numpy()
        label_id = np.argmax(probs)
        
        labels = ['positive', 'negative', 'neutral']
        
        return {
            'label': labels[label_id],
            'score': float(probs[label_id])
        }
    
    def _split_sentences(self, text):
        """Split text into sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _aggregate_results(self, results):
        """Aggregate sentiment results"""
        # Count each sentiment
        positive = sum(1 for r in results if r['label'] == 'positive')
        negative = sum(1 for r in results if r['label'] == 'negative')
        neutral = sum(1 for r in results if r['label'] == 'neutral')
        
        total = len(results)
        
        # Determine dominant sentiment
        if negative > positive and negative > neutral:
            dominant = 'negative'
        elif positive > negative and positive > neutral:
            dominant = 'positive'
        else:
            dominant = 'neutral'
        
        return {
            'sentiment': dominant,
            'positive_ratio': round(positive / total, 3),
            'negative_ratio': round(negative / total, 3),
            'neutral_ratio': round(neutral / total, 3),
            'sentences_analyzed': total
        }
    
    def _empty_result(self):
        """Return empty result"""
        return {
            'sentiment': 'neutral',
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'neutral_ratio': 1.0,
            'sentences_analyzed': 0
        }


# Test the analyzer
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    # Test texts
    print("\n🧪 Testing sentiment analysis:\n")
    
    test_positive = "Our company achieved record profits and strong growth this quarter with excellent performance."
    print("POSITIVE TEXT:")
    result = analyzer.analyze(test_positive)
    print(f"   Result: {result['sentiment']} (confidence: {result[result['sentiment'] + '_ratio']:.2f})")
    
    test_negative = "We face significant risks including lawsuits, market decline, and financial losses. Bankruptcy risk is high."
    print("\nNEGATIVE TEXT:")
    result = analyzer.analyze(test_negative)
    print(f"   Result: {result['sentiment']} (confidence: {result[result['sentiment'] + '_ratio']:.2f})")
