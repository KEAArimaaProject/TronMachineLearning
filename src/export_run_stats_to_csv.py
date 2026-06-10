#!/usr/bin/env python3
"""
Export training metrics (time vs winrate vs greedy) to CSV for Google Sheets.
Usage:
    python export_metrics_to_csv.py --run_dir training_runs/ga_run_20260109_120000 --output metrics.csv
    python export_metrics_to_csv.py --run_dir training_runs/rl_run_20260109_120000 --output metrics.csv
    python export_metrics_to_csv.py --run_dir training_runs --output all_metrics.csv --aggregate
"""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

def extract_ga_metrics(genome_json_path: Path) -> Tuple[float, float]:
    """Extract training_duration_seconds and winrate_vs_greedy from GA genome JSON."""
    with open(genome_json_path, 'r') as f:
        data = json.load(f)
    duration = data.get("training_duration_seconds", 0.0)
    winrate = data.get("winrate_vs_greedy", None)
    if winrate is None:
        # fallback to fitness if winrate missing
        winrate = data.get("fitness", 0.0)
    return float(duration), float(winrate)

def extract_rl_metrics(checkpoint_json_path: Path) -> Tuple[float, float]:
    """Extract training_duration_seconds and winrate_vs_greedy from RL checkpoint JSON."""
    with open(checkpoint_json_path, 'r') as f:
        data = json.load(f)
    duration = data.get("training_duration_seconds", 0.0)
    winrate = data.get("winrate_vs_greedy", None)
    if winrate is None:
        winrate = 0.0
    return float(duration), float(winrate)

def collect_ga_run(run_dir: Path) -> List[Dict]:
    """Collect metrics from all genome JSONs in a GA run."""
    points = []
    genomes_dir = run_dir / "genomes"
    if not genomes_dir.exists():
        print(f"Warning: No genomes directory in {run_dir}")
        return points
    for json_file in sorted(genomes_dir.glob("*.json")):
        # extract generation from filename if possible (optional)
        duration, winrate = extract_ga_metrics(json_file)
        points.append({
            "duration_seconds": duration,
            "winrate_vs_greedy": winrate,
            "source": json_file.name,
            "type": "GA"
        })
    return points

def collect_rl_run(run_dir: Path) -> List[Dict]:
    """Collect metrics from all checkpoint JSONs in an RL run."""
    points = []
    checkpoints_dir = run_dir / "checkpoints_rl"
    if not checkpoints_dir.exists():
        print(f"Warning: No checkpoints_rl directory in {run_dir}")
        return points
    for json_file in sorted(checkpoints_dir.glob("*.json")):
        duration, winrate = extract_rl_metrics(json_file)
        points.append({
            "duration_seconds": duration,
            "winrate_vs_greedy": winrate,
            "source": json_file.name,
            "type": "RL"
        })
    return points

def collect_all_runs(root_dir: Path, aggregate: bool = False) -> List[Dict]:
    """Collect metrics from all GA/RL runs under root_dir."""
    all_points = []
    # Look for directories that contain either 'genomes' or 'checkpoints_rl'
    for subdir in root_dir.iterdir():
        if not subdir.is_dir():
            continue
        # Check if it's a GA run (has genomes folder)
        if (subdir / "genomes").exists():
            points = collect_ga_run(subdir)
            if points:
                # Add run identifier
                for p in points:
                    p["run"] = subdir.name
                all_points.extend(points)
        # Check if it's an RL run
        elif (subdir / "checkpoints_rl").exists():
            points = collect_rl_run(subdir)
            if points:
                for p in points:
                    p["run"] = subdir.name
                all_points.extend(points)
    return all_points

def write_csv(points: List[Dict], output_path: Path):
    """Write points to CSV."""
    if not points:
        print("No data points found. Exiting.")
        return
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ["duration_seconds", "winrate_vs_greedy", "type", "run", "source"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for p in points:
            writer.writerow(p)
    print(f"Saved {len(points)} data points to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to a specific run directory (e.g., training_runs/ga_run_xxx) or root directory containing multiple runs")
    parser.add_argument("--output", type=str, default="metrics.csv",
                        help="Output CSV filename")
    parser.add_argument("--aggregate", action="store_true",
                        help="If set, treat run_dir as root containing multiple runs and aggregate all")
    args = parser.parse_args()

    root = Path(args.run_dir)
    if args.aggregate:
        points = collect_all_runs(root, aggregate=True)
    else:
        # Single run: detect type automatically
        if (root / "genomes").exists():
            points = collect_ga_run(root)
        elif (root / "checkpoints_rl").exists():
            points = collect_rl_run(root)
        else:
            # Try to auto‑detect if it's a run dir with sub‑run folders
            points = collect_all_runs(root, aggregate=True)
    write_csv(points, Path(args.output))

if __name__ == "__main__":
    main()