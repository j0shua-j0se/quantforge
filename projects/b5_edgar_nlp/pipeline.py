"""B5 Main Pipeline - Combines all components"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from edgar_downloader import EDGARDownloader
from text_parser import TextParser
from sentiment_analyzer import SentimentAnalyzer
from risk_scorer import RiskScorer


class B5Pipeline:
    """Complete B5 EDGAR NLP pipeline"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("B5 EDGAR NLP PIPELINE - Initializing")
        print("="*60 + "\n")
        
        self.downloader = EDGARDownloader()
        self.parser = TextParser()
        self.sentiment = SentimentAnalyzer()
        self.scorer = RiskScorer()
        
        print("\n✅ All components loaded\n")
    
    def run(self, tickers, num_filings=2):
        """
        Run complete pipeline for list of tickers
        
        Args:
            tickers: List of ticker symbols
            num_filings: Number of filings per ticker (default: 2)
        
        Returns:
            DataFrame with risk signals
        """
        print("="*60)
        print(f"Processing {len(tickers)} tickers")
        print("="*60 + "\n")
        
        results = []
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] {ticker}")
            print("-" * 40)
            
            # Step 1: Download filings
            files = self.downloader.download_10k(ticker, num_filings)
            
            if not files:
                print(f"   ⚠️  No filings downloaded\n")
                continue
            
            # Step 2: Process each filing
            for file_path in files:
                filing_date = file_path.parent.name[:10]
                
                # Parse Risk Factors
                parsed = self.parser.parse_filing(file_path)
                
                if not parsed['found']:
                    print(f"   ⚠️  {filing_date}: No Risk Factors found")
                    continue
                
                risk_text = parsed['risk_factors']
                print(f"   📄 {filing_date}: Extracted {len(risk_text):,} chars")
                
                # Analyze sentiment
                sentiment_result = self.sentiment.analyze(risk_text, max_sentences=15)
                
                # Calculate risk score
                risk_score = self.scorer.calculate_risk_score(risk_text)
                
                # Store result
                results.append({
                    'ticker': ticker,
                    'filing_date': filing_date,
                    'sentiment': sentiment_result['sentiment'],
                    'positive_ratio': sentiment_result['positive_ratio'],
                    'negative_ratio': sentiment_result['negative_ratio'],
                    'neutral_ratio': sentiment_result['neutral_ratio'],
                    'risk_score': risk_score,
                    'text_length': len(risk_text)
                })
                
                print(f"   📊 Sentiment: {sentiment_result['sentiment']}")
                print(f"   📊 Risk Score: {risk_score}")
            
            print()  # Blank line between tickers
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        if len(df) > 0:
            output_dir = Path("../../data/edgar")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"risk_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(output_file, index=False)
            
            print("="*60)
            print(f"✅ Pipeline Complete!")
            print("="*60)
            print(f"\n📊 Generated {len(df)} risk signals")
            print(f"📁 Saved to: {output_file}")
            print(f"\n📈 Summary Statistics:")
            print(f"   Average Risk Score: {df['risk_score'].mean():.3f}")
            print(f"   Negative Sentiment: {(df['sentiment']=='negative').sum()} filings")
            print(f"   Positive Sentiment: {(df['sentiment']=='positive').sum()} filings")
            print(f"   Neutral Sentiment: {(df['sentiment']=='neutral').sum()} filings")
        else:
            print("⚠️  No risk signals generated")
        
        return df


# Run the pipeline
if __name__ == "__main__":
    pipeline = B5Pipeline()
    
    # Test on 3 tickers
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    risk_signals = pipeline.run(test_tickers, num_filings=2)
    
    if len(risk_signals) > 0:
        print("\n" + "="*60)
        print("SAMPLE RESULTS")
        print("="*60)
        print(risk_signals.to_string(index=False))
