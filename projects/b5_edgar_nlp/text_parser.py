"""Extract sections from 10-K HTML filings"""

import re
from pathlib import Path
from bs4 import BeautifulSoup


class TextParser:
    """Reads 10-K HTML files and extracts key sections"""
    
    def __init__(self):
        self.stats = {
            'total_parsed': 0,
            'risk_factors_found': 0
        }
    
    def parse_filing(self, file_path):
        """
        Extract Risk Factors from a 10-K filing
        
        Args:
            file_path: Path to primary-document.html file
        
        Returns:
            Dictionary with 'risk_factors' text and 'length'
        """
        try:
            # Read the HTML file
            html = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            
            # Parse HTML to plain text
            soup = BeautifulSoup(html, 'lxml')
            full_text = soup.get_text(separator='\n', strip=True)
            
            # Extract Risk Factors section
            risk_factors = self._extract_risk_factors(full_text)
            
            # Update stats
            self.stats['total_parsed'] += 1
            if risk_factors:
                self.stats['risk_factors_found'] += 1
            
            return {
                'risk_factors': risk_factors,
                'length': len(risk_factors),
                'found': bool(risk_factors)
            }
            
        except Exception as e:
            print(f"❌ Failed to parse {Path(file_path).name}: {e}")
            return {
                'risk_factors': '',
                'length': 0,
                'found': False
            }
    
    def _extract_risk_factors(self, text):
        """Find the Risk Factors section in 10-K text"""
        text_lower = text.lower()
        
        # Look for "item 1a" or "risk factors"
        patterns = [
            r'item\s*1a[.\s:]*risk\s*factors',
            r'item\s*1a',
            r'risk\s*factors'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            
            if match:
                start = match.end()
                
                # Take next 20,000 characters (typical Risk Factors length)
                end = start + 20000
                section = text[start:end]
                
                # Clean it up
                section = self._clean_text(section)
                
                # Only return if we got meaningful text
                if len(section) > 500:
                    return section
        
        return ""
    
    def _clean_text(self, text):
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers
        text = re.sub(r'page\s*\d+', '', text, flags=re.IGNORECASE)
        
        # Remove table of contents
        text = re.sub(r'table\s*of\s*contents', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def get_stats(self):
        """Get parsing statistics"""
        return self.stats


# Test the parser
if __name__ == "__main__":
    parser = TextParser()
    
    # Test on downloaded Apple filing
    test_file = Path("sec-edgar-filings/AAPL/10-K/0000320193-24-000123/primary-document.html")
    
    if test_file.exists():
        print(f"📄 Parsing: {test_file.name}")
        result = parser.parse_filing(test_file)
        
        print(f"\n✅ Extraction Result:")
        print(f"   Found: {result['found']}")
        print(f"   Length: {result['length']:,} characters")
        
        if result['risk_factors']:
            print(f"\n🔍 First 500 characters:")
            print(result['risk_factors'][:500])
            print("...")
    else:
        print(f"⚠️  Test file not found: {test_file}")
        print("   Run edgar_downloader.py first to download filings")
