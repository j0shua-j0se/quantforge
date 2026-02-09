"""Data loading, caching, and versioning for reproducible research."""

import pandas as pd
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class DataManager:
    """Manage data snapshots with versioning."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.snapshots_dir = self.data_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.data_dir / "manifest.json"
        self.manifest = self._load_manifest()
    
    def _load_manifest(self) -> Dict:
        """Load existing manifest or create new."""
        if self.manifest_path.exists():
            with open(self.manifest_path, 'r') as f:
                return json.load(f)
        return {}
    
    def save_manifest(self):
        """Save manifest to disk."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)
    
    def save_snapshot(self, df: pd.DataFrame, name: str, description: str = "") -> str:
        """Save data snapshot with versioning."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id = f"{name}_{timestamp}"
        filepath = self.snapshots_dir / f"{snapshot_id}.parquet"
        
        df.to_parquet(filepath)
        
        file_hash = self._compute_hash(filepath)
        
        self.manifest[snapshot_id] = {
            'name': name,
            'timestamp': timestamp,
            'description': description,
            'filepath': str(filepath),
            'rows': len(df),
            'columns': list(df.columns),
            'hash_md5': file_hash,
            'size_mb': filepath.stat().st_size / (1024**2)
        }
        
        self.save_manifest()
        print(f"✓ Snapshot saved: {snapshot_id}")
        return snapshot_id
    
    def load_snapshot(self, snapshot_id: str) -> pd.DataFrame:
        """Load a specific snapshot."""
        if snapshot_id not in self.manifest:
            raise ValueError(f"Snapshot not found: {snapshot_id}")
        
        filepath = self.manifest[snapshot_id]['filepath']
        df = pd.read_parquet(filepath)
        print(f"✓ Loaded snapshot: {snapshot_id} ({len(df)} rows)")
        return df
    
    def list_snapshots(self) -> Dict:
        """List all available snapshots."""
        return self.manifest
    
    @staticmethod
    def _compute_hash(filepath: Path) -> str:
        """Compute MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


def load_sample_data() -> pd.DataFrame:
    """Load sample market data for testing."""
    import yfinance as yf
    
    print("Downloading sample data (SPY 2020-2024)...")
    df = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]
    
    if 'adj_close' in df.columns:
        df = df.rename(columns={'adj_close': 'close'})
    
    print(f"✓ Downloaded {len(df)} days of data")
    return df
