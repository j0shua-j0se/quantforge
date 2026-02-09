"""B1 Data Ingestion - Downloads market data from yFinance and FRED"""
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import List, Tuple
import logging
from fredapi import Fred
import os

logger = logging.getLogger(__name__)


class MarketDataIngestion:
    def __init__(self, config: dict):
        self.config = config
        # Convert string dates to datetime objects
        self.start_date = pd.to_datetime(config['data_sources']['yfinance']['start_date'])
        self.end_date = pd.to_datetime(config['data_sources']['yfinance']['end_date'])
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.fred = Fred(api_key=self.fred_api_key) if self.fred_api_key else None
        
    def get_sp500_tickers(self, limit: int = 100) -> List[str]:
        """Get SP500 ticker list from Wikipedia"""
        logger.info("Fetching SP500 tickers...")
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            tickers = tables[0]['Symbol'].tolist()
            tickers = [t.replace('.', '-') for t in tickers]
            logger.info(f"✓ Found {len(tickers)} tickers")
            return tickers[:limit]
        except Exception as e:
            logger.warning(f"Wikipedia failed, using backup list: {e}")
            return [
    # Mega Cap (10)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'JPM', 'V',
    # Large Cap Tech (20)
    'MA', 'NFLX', 'CSCO', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'QCOM', 'TXN',
    'AVGO', 'IBM', 'NOW', 'INTU', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS',
    # Finance (15)
    'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'SPGI', 'CME',
    'USB', 'PNC', 'TFC', 'COF', 'BK',
    # Healthcare (15)
    'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'LLY', 'MRK', 'DHR', 'BMY',
    'AMGN', 'GILD', 'CVS', 'CI', 'HUM',
    # Consumer (15)
    'WMT', 'HD', 'MCD', 'NKE', 'COST', 'SBUX', 'TGT', 'LOW', 'TJX', 'DG',
    'CMG', 'YUM', 'ROST', 'ULTA', 'BURL',
    # Industrials (10)
    'BA', 'CAT', 'DE', 'UPS', 'HON', 'UNP', 'RTX', 'LMT', 'GE', 'MMM',
    # Energy (5)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    # Consumer Staples (5)
    'PG', 'KO', 'PEP', 'PM', 'MDLZ',
    # Utilities & Telecom (5)
    'NEE', 'DUK', 'SO', 'T', 'VZ'
][:limit]  # Return only the requested limit

    
    def download_market_data(self, tickers: List[str] = None) -> pd.DataFrame:
        """Download OHLCV data from yfinance"""
        if tickers is None:
            tickers = self.get_sp500_tickers(limit=10)
        
        logger.info(f"📥 Downloading {len(tickers)} tickers from {self.start_date.date()} to {self.end_date.date()}...")
        all_data = []
        failed_tickers = []
        
        for i, ticker in enumerate(tickers, 1):
            try:
                logger.info(f"  [{i}/{len(tickers)}] {ticker}")
                
                # Download using Ticker with datetime objects
                stock = yf.Ticker(ticker)
                data = stock.history(start=self.start_date, end=self.end_date)
                
                if len(data) == 0:
                    logger.warning(f"  ⚠ {ticker}: No data returned")
                    failed_tickers.append(ticker)
                    continue
                
                # Add ticker column
                data['ticker'] = ticker
                data['date'] = data.index
                data = data.reset_index(drop=True)
                
                # Standardize column names
                data.columns = [col.lower().replace(' ', '_') for col in data.columns]
                
                all_data.append(data)
                logger.info(f"  ✓ {ticker}: {len(data)} records")
                
            except Exception as e:
                logger.error(f"  ✗ {ticker}: {e}")
                failed_tickers.append(ticker)
        
        if not all_data:
            raise ValueError("No data downloaded! Check your internet connection.")
        
        market_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"✓ Downloaded {len(market_data)} total records from {len(all_data)} tickers")
        
        if failed_tickers:
            logger.warning(f"⚠ Failed tickers: {', '.join(failed_tickers)}")
        
        return market_data
    
    def download_macro_data(self) -> pd.DataFrame:
        """Download macro data from FRED"""
        if not self.fred:
            logger.warning("FRED API not available")
            return pd.DataFrame()
        
        logger.info("📥 Downloading FRED data...")
        macro_data = {}
        
        # Convert start date to string for FRED API
        start_str = self.start_date.strftime('%Y-%m-%d')
        end_str = self.end_date.strftime('%Y-%m-%d')
        
        for series in self.config['data_sources']['fred']['series']:
            try:
                data = self.fred.get_series(
                    series['code'], 
                    observation_start=start_str,
                    observation_end=end_str
                )
                macro_data[series['name']] = data
                logger.info(f"  ✓ {series['name']}: {len(data)} records")
            except Exception as e:
                logger.error(f"  ✗ {series['name']}: {e}")
        
        if macro_data:
            df = pd.DataFrame(macro_data)
            df['date'] = df.index
            df = df.reset_index(drop=True)
            logger.info(f"✓ Downloaded {len(df)} macro records")
            return df
        
        return pd.DataFrame()
    
    def save_raw_snapshots(self, market_data: pd.DataFrame, macro_data: pd.DataFrame, 
                          version_date: str) -> Tuple[str, str]:
        """Save raw data snapshots"""
        os.makedirs('data/market', exist_ok=True)
        os.makedirs('data/macro', exist_ok=True)
        
        market_path = f"data/market/market_raw_{version_date}.parquet"
        macro_path = f"data/macro/macro_raw_{version_date}.parquet"
        
        market_data.to_parquet(market_path, compression='snappy', index=False)
        logger.info(f"  💾 Market: {market_path}")
        
        if len(macro_data) > 0:
            macro_data.to_parquet(macro_path, compression='snappy', index=False)
            logger.info(f"  💾 Macro: {macro_path}")
        
        logger.info(f"✓ Saved raw snapshots: {version_date}")
        return market_path, macro_path
