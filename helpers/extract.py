import os
import json
import csv

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "experiments", "results")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "metrics_summary.csv")

METRIC_KEYS = [
    "avg_original_score",
    "max_original_score",
    "final_avg_original",
    "score_improvement",
    "top_10_percent_avg",
    "avg_shaped_reward",
    "avg_episode_length",
    "max_episode_length",
    "length_improvement",
]

def iter_result_files(root_dir):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".json"):
                yield os.path.join(root, f)

def load_json(path):
    with open(path, "r") as fp:
        return json.load(fp)

def extract_metrics(obj):
    metrics = obj.get("metrics", {})
    return {k: metrics.get(k, None) for k in METRIC_KEYS}

def main():
    rows = []
    files = list(iter_result_files(RESULTS_DIR))
    if not files:
        print(f"No JSON files found under: {RESULTS_DIR}")
        return

    print(f"Found {len(files)} JSON file(s). Extracting metrics...\n")

    for path in files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"Skipping {path}: failed to parse JSON ({e})")
            continue

        run_name = (data.get("config", {}) or {}).get("name") or os.path.splitext(os.path.basename(path))[0]
        metrics = extract_metrics(data)

        # Print to console
        print(f"- {run_name} ({os.path.relpath(path, RESULTS_DIR)})")
        for k in METRIC_KEYS:
            print(f"  {k}: {metrics.get(k)}")
        print("")

        # Collect for CSV
        row = {"run_name": run_name, "file": os.path.relpath(path, RESULTS_DIR)}
        row.update(metrics)
        rows.append(row)

    # Write CSV summary
    fieldnames = ["run_name", "file"] + METRIC_KEYS
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote CSV summary: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()