"""Download SEC 10-K filings"""

import os
import time
import logging
from pathlib import Path
from sec_edgar_downloader import Downloader

logger = logging.getLogger(__name__)


class EDGARDownloader:
    """Downloads 10-K filings from SEC EDGAR"""
    
    def __init__(self, company_name="QuantForge Research", email="joshua.jose2002@gmail.com"):
        """Initialize downloader"""
        self.downloader = Downloader(company_name, email)
        self.cache_dir = Path("../../data/edgar/raw")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ EDGAR Downloader ready")
        print(f"   Cache: {self.cache_dir.absolute()}")
    
    def download_10k(self, ticker, num_filings=3):
        """
        Download 10-K filings for one ticker
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            num_filings: Number of filings to download (default: 3)
        
        Returns:
            List of file paths to downloaded filings
        """
        print(f"\n⬇️  Downloading {num_filings} 10-K filings for {ticker}...")
        
        try:
            # Download from SEC
            self.downloader.get(
                "10-K",
                ticker,
                limit=num_filings,
                after="2020-01-01",
                download_details=True
            )
            
            # Find downloaded files
            filing_dir = Path(f"sec-edgar-filings/{ticker}/10-K")
            if not filing_dir.exists():
                print(f"❌ {ticker}: No filings found")
                return []
            
            # Get all primary-document.html files
            files = list(filing_dir.glob("*/primary-document.html"))
            
            print(f"✅ {ticker}: Downloaded {len(files)} filings")
            
            # Wait to respect SEC rate limit (10 requests/second max)
            time.sleep(0.15)
            
            return files
            
        except Exception as e:
            print(f"❌ {ticker}: Failed - {e}")
            return []
    
    def download_batch(self, tickers, num_filings=3):
        """
        Download 10-Ks for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            num_filings: Number of filings per ticker
        
        Returns:
            Dictionary mapping ticker -> list of file paths
        """
        results = {}
        
        print(f"\n{'='*60}")
        print(f"Downloading 10-Ks for {len(tickers)} tickers")
        print(f"{'='*60}")
        
        for i, ticker in enumerate(tickers, 1):
            print(f"\n[{i}/{len(tickers)}] {ticker}")
            files = self.download_10k(ticker, num_filings)
            results[ticker] = files
        
        total_files = sum(len(files) for files in results.values())
        print(f"\n✅ Total: Downloaded {total_files} filings for {len(tickers)} tickers")
        
        return results


# Test the downloader
if __name__ == "__main__":
    # Create downloader
    downloader = EDGARDownloader()
    
    # Test on Apple
    files = downloader.download_10k("AAPL", num_filings=2)
    
    print(f"\n📁 Downloaded files:")
    for f in files:
        print(f"   {f}")
