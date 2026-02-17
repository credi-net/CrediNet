import pandas as pd
from pathlib import Path

def main():
    domains_set = set()

    for split_type in ['train', 'val', 'test']:
        file = f"data/all_splits/regression/{split_type}_regression_domains.parquet"
        if Path(file).exists():
            df = pd.read_parquet(file)
            domains = df['domain'].tolist()
            domains_set.update(domains)
            print(f"Added {len(domains)} domains from {file}")

    for split_type in ['train', 'val', 'test']:
        file = f"data/all_splits/binary/{split_type}_domains.parquet"
        if Path(file).exists():
            df = pd.read_parquet(file)
            domains = df['domain'].tolist()
            domains_set.update(domains)
            print(f"Added {len(domains)} domains from {file}")

    output_file = "data/all_domains_for_features.csv"
    df_domains = pd.DataFrame({'domain': sorted(list(domains_set))})
    df_domains.to_csv(output_file, index=False)

    print(f"\nTotal unique domains: {len(domains_set)}")
    print(f"Saved to: {output_file}")

if __name__=="__main__": 
    main()