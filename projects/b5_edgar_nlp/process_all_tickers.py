"""Process all QuantForge tickers through B5 pipeline"""

import pandas as pd
from pathlib import Path
from pipeline import B5Pipeline
import time

# Load your tickers from B1 market data
print("Loading tickers from B1 data...")

market_data_file = Path("../../data/market/market_clean_20260209.parquet")

if not market_data_file.exists():
    print("❌ Market data not found!")
    exit(1)

print(f"✅ Using: {market_data_file.name}")
market_data = pd.read_parquet(market_data_file)
all_tickers = sorted(market_data['ticker'].unique().tolist())

print(f"\n📊 Found {len(all_tickers)} tickers")
print(f"   Sample: {all_tickers[:10]}")

# Ask for confirmation
print(f"\n⚠️  This will download ~{len(all_tickers) * 2} 10-K filings")
print(f"   Estimated time: {len(all_tickers) * 1.5:.0f} minutes (~{len(all_tickers) * 1.5 / 60:.1f} hours)")
print(f"\n   You can stop anytime with Ctrl+C and resume later")

response = input("\nProceed? (yes/no): ")

if response.lower() != 'yes':
    print("Cancelled.")
    exit(0)

# Initialize pipeline
pipeline = B5Pipeline()

# Process in batches of 10 (easier to track progress)
batch_size = 10
all_results = []

start_time = time.time()

for batch_num in range(0, len(all_tickers), batch_size):
    batch_tickers = all_tickers[batch_num:batch_num + batch_size]
    batch_label = f"Batch {batch_num//batch_size + 1}/{(len(all_tickers)-1)//batch_size + 1}"
    
    print(f"\n{'='*60}")
    print(f"{batch_label}: Processing {len(batch_tickers)} tickers")
    print(f"{'='*60}")
    
    try:
        # Process batch
        batch_results = pipeline.run(batch_tickers, num_filings=2)
        
        if len(batch_results) > 0:
            all_results.append(batch_results)
        
        # Progress update
        elapsed = time.time() - start_time
        processed = batch_num + len(batch_tickers)
        rate = processed / elapsed * 60  # tickers per minute
        remaining = (len(all_tickers) - processed) / rate
        
        print(f"\n⏱️  Progress: {processed}/{len(all_tickers)} tickers")
        print(f"   Elapsed: {elapsed/60:.1f} min | Remaining: ~{remaining:.1f} min")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user")
        print(f"   Processed: {batch_num}/{len(all_tickers)} tickers")
        break
    
    except Exception as e:
        print(f"\n❌ Batch failed: {e}")
        print("   Continuing with next batch...")

# Combine all results
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save final results
    output_file = Path("../../data/edgar/risk_signals_ALL_20260212.csv")
    final_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Total risk signals: {len(final_df)}")
    print(f"📁 Saved to: {output_file}")
    print(f"\n📈 Final Statistics:")
    print(f"   Companies processed: {final_df['ticker'].nunique()}")
    print(f"   Average risk score: {final_df['risk_score'].mean():.3f}")
    print(f"   High risk (>0.6): {(final_df['risk_score'] > 0.6).sum()} filings")
    print(f"   Low risk (<0.4): {(final_df['risk_score'] < 0.4).sum()} filings")
    
    # Show top 10 riskiest companies
    print(f"\n🔴 Top 10 Riskiest Companies:")
    top_risk = final_df.groupby('ticker')['risk_score'].mean().sort_values(ascending=False).head(10)
    for ticker, score in top_risk.items():
        print(f"   {ticker}: {score:.3f}")
    
    # Show top 10 safest companies
    print(f"\n🟢 Top 10 Safest Companies:")
    low_risk = final_df.groupby('ticker')['risk_score'].mean().sort_values().head(10)
    for ticker, score in low_risk.items():
        print(f"   {ticker}: {score:.3f}")
else:
    print("\n❌ No results generated")

total_time = time.time() - start_time
print(f"\n⏱️  Total time: {total_time/60:.1f} minutes")
