from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class CredStats:
    total: int = 0
    with_cred: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_val: float = math.inf
    max_val: float = -math.inf

    def update(self, cred: Optional[float]) -> None:
        self.total += 1
        if cred is None:
            return
        self.with_cred += 1
        delta = cred - self.mean
        self.mean += delta / self.with_cred
        delta2 = cred - self.mean
        self.m2 += delta * delta2
        if cred < self.min_val:
            self.min_val = cred
        if cred > self.max_val:
            self.max_val = cred

    def variance(self) -> Optional[float]:
        if self.with_cred < 2:
            return None
        return self.m2 / (self.with_cred - 1)

    def stddev(self) -> Optional[float]:
        var = self.variance()
        if var is None:
            return None
        return math.sqrt(var)

    def range(self) -> Optional[float]:
        if self.with_cred == 0:
            return None
        return self.max_val - self.min_val

    def missing_pct(self) -> Optional[float]:
        if self.total == 0:
            return None
        return 1.0 - (self.with_cred / self.total)


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        num = float(value)
    except ValueError:
        return None
    if not math.isfinite(num):
        return None
    return num


def read_rows(path: str) -> Iterable[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def analyze_rows(
    rows: Iterable[Dict[str, str]],
) -> Tuple[Dict[str, CredStats], CredStats, Dict[str, int], List[float]]:
    analysis_stats: Dict[str, CredStats] = {}
    overall = CredStats()
    counts = {
        "rows": 0,
        "unique_sources": 0,
        "unique_domains": 0,
    }

    source_ids = set()
    domains = set()
    domains_with_cred = set()

    cred_values: List[float] = []

    for row in rows:
        counts["rows"] += 1
        analysis_id = (row.get("analysis_id") or "").strip()
        if analysis_id == "":
            analysis_id = "<missing>"
        if analysis_id not in analysis_stats:
            analysis_stats[analysis_id] = CredStats()
        cred = parse_float(row.get("domain_credibility", ""))
        analysis_stats[analysis_id].update(cred)
        overall.update(cred)
        if cred is not None:
            cred_values.append(cred)

        source_id = (row.get("source_id") or "").strip()
        if source_id != "":
            source_ids.add(source_id)
        domain = (row.get("domain_name") or "").strip()
        if domain != "":
            domains.add(domain)
            if cred is not None:
                domains_with_cred.add(domain)

    counts["unique_sources"] = len(source_ids)
    counts["unique_domains"] = len(domains)
    counts["domains_with_cred"] = len(domains_with_cred)
    return analysis_stats, overall, counts, cred_values


def analyze(
    path: str,
) -> Tuple[Dict[str, CredStats], CredStats, Dict[str, int], List[float]]:
    return analyze_rows(read_rows(path))


def write_per_analysis_csv(path: str, stats: Dict[str, CredStats]) -> None:
    fieldnames = [
        "analysis_id",
        "sources_total",
        "sources_with_cred",
        "cred_mean",
        "cred_variance",
        "cred_stddev",
        "cred_min",
        "cred_max",
        "cred_range",
        "cred_missing_pct",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for analysis_id, stat in stats.items():
            writer.writerow(
                {
                    "analysis_id": analysis_id,
                    "sources_total": stat.total,
                    "sources_with_cred": stat.with_cred,
                    "cred_mean": stat.mean if stat.with_cred else "",
                    "cred_variance": stat.variance() if stat.with_cred else "",
                    "cred_stddev": stat.stddev() if stat.with_cred else "",
                    "cred_min": stat.min_val if stat.with_cred else "",
                    "cred_max": stat.max_val if stat.with_cred else "",
                    "cred_range": stat.range() if stat.with_cred else "",
                    "cred_missing_pct": stat.missing_pct(),
                }
            )


def top_variance(stats: Dict[str, CredStats], top_n: int, min_with_cred: int) -> list:
    ranked = []
    for analysis_id, stat in stats.items():
        if stat.with_cred < min_with_cred:
            continue
        var = stat.variance()
        if var is None:
            continue
        ranked.append((analysis_id, var, stat))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_n]


def percentile(sorted_values: List[float], pct: float) -> Optional[float]:
    if not sorted_values:
        return None
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    idx = (len(sorted_values) - 1) * (pct / 100.0)
    lower = math.floor(idx)
    upper = math.ceil(idx)
    if lower == upper:
        return sorted_values[int(idx)]
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def build_summary(
    file_path: str,
    stats: Dict[str, CredStats],
    overall: CredStats,
    counts: Dict[str, int],
    cred_values: List[float],
) -> Dict[str, object]:
    file_size = os.path.getsize(file_path)
    analyses = len(stats)
    with_cred = sum(1 for s in stats.values() if s.with_cred > 0)
    cred_sorted = sorted(cred_values)
    return {
        "file_path": file_path,
        "file_size_bytes": file_size,
        "rows": counts["rows"],
        "unique_analysis_ids": analyses,
        "unique_sources": counts["unique_sources"],
        "unique_domains": counts["unique_domains"],
        "analysis_ids_with_any_cred": with_cred,
        "domains_with_cred": counts["domains_with_cred"],
        "overall_cred_values": overall.with_cred,
        "overall_cred_mean": overall.mean if overall.with_cred else None,
        "overall_cred_variance": overall.variance(),
        "overall_cred_stddev": overall.stddev(),
        "overall_cred_min": overall.min_val if overall.with_cred else None,
        "overall_cred_max": overall.max_val if overall.with_cred else None,
        "overall_cred_q25": percentile(cred_sorted, 25),
        "overall_cred_q50": percentile(cred_sorted, 50),
        "overall_cred_q75": percentile(cred_sorted, 75),
    }


def print_summary(summary: Dict[str, object]) -> None:
    logging.info("File size (bytes): %s", summary["file_size_bytes"])
    logging.info("Rows: %s", summary["rows"])
    logging.info("Unique analysis_id: %s", summary["unique_analysis_ids"])
    logging.info("Unique sources: %s", summary["unique_sources"])
    logging.info("Unique domains: %s", summary["unique_domains"])
    logging.info("Analysis with any credibility: %s", summary["analysis_ids_with_any_cred"])
    logging.info("Cred values total: %s", summary["overall_cred_values"])
    logging.info("Cred mean: %s", summary["overall_cred_mean"])
    logging.info("Cred variance: %s", summary["overall_cred_variance"])
    logging.info("Cred stddev: %s", summary["overall_cred_stddev"])
    logging.info("Cred min/max: %s / %s", summary["overall_cred_min"], summary["overall_cred_max"])
    logging.info("Domains with credibility: %s", summary["domains_with_cred"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze user_generated_data_with_sources.csv with streaming stats"
    )
    parser.add_argument(
        "--input",
        default="data/user_generated_data_with_sources.csv",
        help="Path to user_generated_data_with_sources.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="output_dir/user_generated_analysis",
        help="Output directory for summaries",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top N analysis_id by credibility variance",
    )
    parser.add_argument(
        "--min-cred-sources",
        type=int,
        default=2,
        help="Minimum sources with credibility to compute variance",
    )
    
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, "INFO"), format="%(message)s")

    os.makedirs(args.out_dir, exist_ok=True)

    stats, overall, counts, cred_values = analyze(args.input)
    summary = build_summary(args.input, stats, overall, counts, cred_values)
    print_summary(summary)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    per_analysis_path = os.path.join(args.out_dir, "per_analysis_cred_stats.csv")
    write_per_analysis_csv(per_analysis_path, stats)

    top_path = os.path.join(args.out_dir, "top_variance.csv")
    top = top_variance(stats, args.top_n, args.min_cred_sources)
    with open(top_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "analysis_id",
                "cred_variance",
                "cred_stddev",
                "sources_with_cred",
                "cred_min",
                "cred_max",
                "cred_range",
            ]
        )
        for analysis_id, var, stat in top:
            writer.writerow(
                [
                    analysis_id,
                    var,
                    stat.stddev(),
                    stat.with_cred,
                    stat.min_val,
                    stat.max_val,
                    stat.range(),
                ]
            )

    logging.info("Wrote summary to: %s", summary_path)
    logging.info("Wrote per-analysis stats to: %s", per_analysis_path)
    logging.info("Wrote top variance list to: %s", top_path)


if __name__ == "__main__":
    main()
