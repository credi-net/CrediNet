"""
Stream graph data from HuggingFace and compute PageRank + graph features for domains.
"""

import gzip
import csv
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_url
import urllib.request
import io
from credigraph.utils.domain_handler import flip_domain

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingGraphProcessor:
    """Stream and process graph data from HuggingFace."""
    
    def __init__(self, month: str, test_domains_csv: str, output_dir: str):
        self.month = month
        self.test_domains_csv = Path(test_domains_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_domains: Set[str] = set()
        self.domain_mapping: Dict[str, int] = {}
        self.indeg: Dict[str, int] = defaultdict(int)
        self.outdeg: Dict[str, int] = defaultdict(int)
        self.outneighbors: Dict[str, Set[str]] = defaultdict(set)
        self.inneighbors: Dict[str, Set[str]] = defaultdict(set)
    
    def load_test_domains(self):
        """Load test set domains and normalize them."""
        from credigraph.utils.domain_handler import flip_if_needed, normalize_domain
        
        logger.info(f"Loading test domains from {self.test_domains_csv}")
        df = pd.read_csv(self.test_domains_csv)
        
        raw_domains = df['domain'].astype(str).tolist()
        normalized_domains = {normalize_domain(flip_if_needed(d)) for d in raw_domains}
        self.test_domains = {flip_domain(d) for d in normalized_domains}
        
        logger.info(f"Loaded {len(raw_domains)} domains, normalized to {len(self.test_domains)} unique reversed domains")
        logger.info(f"Example: {list(self.test_domains)[:3]}")
    
    def stream_graph_data(self):
        """Stream and process graph data from HuggingFace."""
        from huggingface_hub import hf_hub_url
        
        logger.info(f"Streaming graph data for {self.month} from HuggingFace...")
        
        edges_url = hf_hub_url(
            repo_id="credi-net/CrediBench",
            filename=f"{self.month}/edges.csv.gz",
            repo_type="dataset"
        )
        vertices_url = hf_hub_url(
            repo_id="credi-net/CrediBench",
            filename=f"{self.month}/vertices.csv.gz",
            repo_type="dataset"
        )
        
        logger.info("Processing vertices...")
        with urllib.request.urlopen(vertices_url) as response:
            with gzip.GzipFile(fileobj=response) as gz_file:
                reader = csv.DictReader(io.TextIOWrapper(gz_file))
                count = 0
                for row in reader:
                    domain = row['domain'].lower()
                    if domain in self.test_domains:
                        out_deg = int(row.get('out_deg', 0))
                        self.outdeg[domain] = out_deg
                    count += 1
                    if count % 100_000 == 0:
                        logger.info(f"  Processed {count:,} vertices")
        
        logger.info("Processing edges...")
        with urllib.request.urlopen(edges_url) as response:
            with gzip.GzipFile(fileobj=response) as gz_file:
                reader = csv.DictReader(io.TextIOWrapper(gz_file))
                count = 0
                for row in reader:
                    src = row['src'].lower()
                    dst = row['dst'].lower()
                    
                    if src in self.test_domains or dst in self.test_domains:
                        self.outneighbors[src].add(dst)
                        self.inneighbors[dst].add(src)
                        if dst in self.test_domains:
                            self.indeg[dst] += 1
                    
                    count += 1
                    if count % 1_000_000 == 0:
                        logger.info(f"  Processed {count:,} edges")
        
        logger.info(f"Processed {count:,} total edges")
    
    def compute_features(self) -> pd.DataFrame:
        """Compute graph features for test domains."""
        logger.info("Computing graph features...")
        
        features = []
        for domain in sorted(self.test_domains):
            indeg = self.indeg.get(domain, 0)
            outdeg = self.outdeg.get(domain, 0)
            
            unique_in = len(self.inneighbors.get(domain, set()))
            unique_out = len(self.outneighbors.get(domain, set()))
            
            # Simple PageRank proxy: (indeg + 1) / (total_domains)
            page_rank = (indeg + 1) / (len(self.test_domains) + 1)
            
            # Rank: lower rank = more authoritative (more incoming links)
            rank = len([d for d in self.test_domains if self.indeg.get(d, 0) > indeg]) + 1
            
            features.append({
                'url': domain,
                'page_rank_decimal': page_rank,
                'rank': rank,
                'root_domains_to_root_domain': unique_in,
                'root_domains_from_root_domain': unique_out,
            })
        
        df = pd.DataFrame(features)
        logger.info(f"Computed features for {len(df)} domains")
        return df
    
    def save_features(self, df: pd.DataFrame):
        """Save features to CSV."""
        output_path = self.output_dir / f"{self.month}_domain_features.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved features to {output_path}")
        return output_path
    
    def run(self):
        """Execute the full pipeline."""
        logger.info("=" * 80)
        logger.info(f"Graph Feature Computation - {self.month}")
        logger.info("=" * 80)
        
        self.load_test_domains()
        self.stream_graph_data()
        features_df = self.compute_features()
        self.save_features(features_df)
        
        logger.info("=" * 80)
        logger.info("Pipeline complete!")
        logger.info("=" * 80)
        
        return features_df


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <month> <test_domains.csv> <output_dir>")
        print(f"  month: oct2024, nov2024, or dec2024")
        sys.exit(1)
    
    month = sys.argv[1]
    test_domains_csv = sys.argv[2]
    output_dir = sys.argv[3]
    
    processor = StreamingGraphProcessor(month, test_domains_csv, output_dir)
    processor.run()
