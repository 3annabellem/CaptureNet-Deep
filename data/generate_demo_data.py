#!/usr/bin/env python3
"""
Generate realistic synthetic nanopore 'current' signal (single channel),
save in repo-friendly format, and display a quick plot.

Outputs:
- data/demo_signal.csv           # column 'current'
- data/demo_labels.json          # [{"start": int, "end": int}, ...]
- data/demo_dataset.csv          # raw_signal path + JSON label + run_id

Usage:
  python data/generate_demo_data.py
  # or with options
  python data/generate_demo_data.py --length 60000 --seed 42 --out-prefix demo
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ar1_noise(n, sigma=1.5, rho=0.98, seed=None):
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=n)
    x = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        x[i] = rho * x[i-1] + eps[i]
    return x

def lowfreq_trend(n, amp=3.0, cycles=1.5, seed=None):
    # slow drift + very low frequency wobble
    t = np.linspace(0, 2*np.pi*cycles, n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2*np.pi)
    return amp * np.sin(t + phase)

def random_spikes(n, count=120, spike_amp=8.0, width=1, seed=None):
    # occasional brief blips during open pore (positive or negative)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n - width, size=count)
    s = np.zeros(n, dtype=np.float32)
    signs = rng.choice([-1.0, 1.0], size=count)
    for i, sign in zip(idx, signs):
        s[i:i+width] += sign * spike_amp
    return s

def place_captures(n, k=3, min_len=3500, max_len=7000, guard=2000, seed=None):
    """
    Choose k non-overlapping [start,end) capture intervals.
    Ensures each is within bounds with a guard region from edges and each other.
    """
    rng = np.random.default_rng(seed)
    intervals = []
    attempts = 0
    while len(intervals) < k and attempts < 1000:
        attempts += 1
        length = int(rng.integers(min_len, max_len))
        start = int(rng.integers(guard, n - guard - length))
        end = start + length
        # Check overlap
        if any(not (end + guard <= s or e + guard <= start) for s, e in intervals):
            continue
        intervals.append([start, end])
    intervals.sort(key=lambda x: x[0])
    return intervals

def synth_signal(length=60000, seed=42):
    """
    Build a realistic single-channel current trace:
    - baseline ~100 pA with AR(1) noise
    - slow drift
    - random spikes
    - capture intervals ~20 pA, noisy with occasional near-zero dips
    """
    rng = np.random.default_rng(seed)
    base = 100.0

    # Background (open pore) = baseline + AR(1) + slow drift + sporadic spikes
    x = base + ar1_noise(length, sigma=2.5, rho=0.985, seed=seed) \
            + lowfreq_trend(length, amp=2.0, cycles=1.0, seed=seed) \
            + random_spikes(length, count=100, spike_amp=6.0, width=1, seed=seed)

    # Dead/closed pore: optional short flat segments (rare)
    # (We’ll skip explicit dead segments; captures provide the key structure.)

    # Inject capture intervals
    caps = place_captures(length, k=3, min_len=3500, max_len=7000, guard=2500, seed=seed)
    for (s, e) in caps:
        # Lower current ~20 pA with smaller noise and intermittent sharp drops ~0 pA
        cap_len = e - s
        cap_noise = np.random.default_rng(seed+1).normal(0, 1.2, size=cap_len).astype(np.float32)
        cap = 20.0 + cap_noise

        # Insert rare near-zero dips within capture to mimic motor stalls / blockades
        dips = np.random.default_rng(seed+2).integers(s, e, size=max(1, cap_len // 800))
        for d in dips:
            width = int(np.random.default_rng(seed+3).integers(2, 6))
            d_end = min(e, d + width)
            cap[d - s:d_end - s] = np.random.default_rng(seed+4).normal(1.5, 0.8, size=d_end - d)

        x[s:e] = cap

    return x.astype(np.float32), caps

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-prefix", type=str, default="demo", help="Filename prefix for outputs")
    ap.add_argument("--out-dir", type=str, default="data")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    signal, captures = synth_signal(length=args.length, seed=args.seed)

    # Save CSV (column 'current')
    csv_path = os.path.join(args.out_dir, f"{args.out_prefix}_signal.csv")
    pd.DataFrame({"current": signal}).to_csv(csv_path, index=False)

    # Save labels JSON
    labels = [{"start": int(s), "end": int(e)} for s, e in captures]
    lbl_path = os.path.join(args.out_dir, f"{args.out_prefix}_labels.json")
    with open(lbl_path, "w") as f:
        json.dump(labels, f)

    # Dataset CSV (raw_signal path + label JSON + run_id)
    ds_path = os.path.join(args.out_dir, f"{args.out_prefix}_dataset.csv")
    ds = pd.DataFrame({
        "raw_signal": [csv_path],
        "label": [json.dumps(labels)],
        "run_id": [f"{args.out_prefix}_run"]
    })
    ds.to_csv(ds_path, index=False)

    print(f"Wrote:\n  {csv_path}\n  {lbl_path}\n  {ds_path}")
    print(f"Intervals (samples): {labels}")

    # Plot
    start, end = 0, len(signal)
    plt.figure(figsize=(10, 3.0))
    plt.plot(np.arange(start, end), signal[start:end])
    for s, e in captures:
        # only shade region if it overlaps the viewed segment
        a, b = max(s, start), min(e, end)
        if a < b:
            plt.axvspan(a - start, b - start, alpha=0.15)
    plt.title("Synthetic Channel Current")
    plt.xlabel("Sample index")
    plt.ylabel("Current (pA)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
