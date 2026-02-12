"""Calculate risk score from text based on keywords"""


class RiskScorer:
    """Scores text based on financial risk keywords"""
    
    def __init__(self):
        # Risk keywords with weights (from config)
        self.risk_words = {
            'risk': 1.0,
            'loss': 0.9,
            'decline': 0.8,
            'lawsuit': 0.9,
            'litigation': 0.85,
            'uncertainty': 0.7,
            'bankruptcy': 1.0,
            'default': 0.9,
            'adverse': 0.8,
            'volatile': 0.7
        }
    
    def calculate_risk_score(self, text):
        """
        Calculate risk score (0 to 1, higher = more risky)
        
        Args:
            text: Text to analyze
        
        Returns:
            Float between 0.0 and 1.0
        """
        if len(text) < 100:
            return 0.0
        
        text_lower = text.lower()
        
        # Count each risk word
        total_score = 0.0
        word_counts = {}
        
        for word, weight in self.risk_words.items():
            count = text_lower.count(word)
            word_counts[word] = count
            total_score += count * weight
        
        # Normalize by text length (per 1000 characters)
        risk_score = total_score / (len(text) / 1000)
        
        # Cap at 1.0
        risk_score = min(1.0, risk_score)
        
        return round(risk_score, 3)
    
    def get_top_risk_words(self, text, top_n=5):
        """Get the most frequent risk words in text"""
        text_lower = text.lower()
        
        word_counts = {}
        for word in self.risk_words.keys():
            count = text_lower.count(word)
            if count > 0:
                word_counts[word] = count
        
        # Sort by count
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_words[:top_n]


# Test the scorer
if __name__ == "__main__":
    scorer = RiskScorer()
    
    print("🧪 Testing risk scoring:\n")
    
    low_risk = "The company operates in a stable market with consistent profits and strong performance."
    print("LOW RISK TEXT:")
    score = scorer.calculate_risk_score(low_risk)
    print(f"   Score: {score}")
    
    high_risk = """We face significant risk from litigation and bankruptcy risk. 
    Market decline and financial losses are expected. Lawsuit outcomes remain uncertain. 
    Default risk is high due to adverse regulatory changes and volatile market conditions. 
    Additional risks include litigation expenses and potential bankruptcy filing."""
    
    print("\nHIGH RISK TEXT:")
    score = scorer.calculate_risk_score(high_risk)
    print(f"   Score: {score}")
    
    top_words = scorer.get_top_risk_words(high_risk)
    print(f"   Top risk words: {top_words}")
